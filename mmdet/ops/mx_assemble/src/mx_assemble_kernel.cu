#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <THC/THCAtomics.cuh>

#include <stdio.h>
#include <math.h>
#include <float.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

#define ROUND_OFF 50000
#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32
#define kMaxThreadsPerBlock 1024

// == Correlation Kernel
template <typename Dtype>
__global__ void CorrelateData(const int nthreads, int num, int topwidth,
  int topheight, int topchannels, int topcount,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int kernel_size, int stride1, int stride2,
  int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *bottom0, const Dtype *bottom1, Dtype *top) {
  extern __shared__ char patch_data_char[];
  Dtype *patch_data = reinterpret_cast<Dtype *>(patch_data_char);
  //  First (upper left) position of kernel upper-left corner
  //  in current center position of neighborhood in image 1
  int x1 = blockIdx.x * stride1 + max_displacement;
  int y1 = blockIdx.y * stride1 + max_displacement;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;
  //  Load 3D patch into shared shared memory
  for (int j = 0; j < kernel_size; j++) {  //  HEIGHT
    for (int i = 0; i < kernel_size; i++) {  //  WIDTH
      int ji_off = ((j * kernel_size) + i) * bottomchannels;
      for (int ch = ch_off; ch < bottomchannels; ch += (THREADS_PER_WARP * WARPS_PER_BLOCK))  {
          //  CHANNELS
          int idx1 = ((item * bottomheight + y1+j) * bottomwidth + x1+i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = bottom0[idx1];
      }
    }
  }
  __syncthreads();
  __shared__ Dtype sum[THREADS_PER_WARP * WARPS_PER_BLOCK];
  //  Compute correlation
  for (int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;
    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    for (int j = 0; j < kernel_size; j++) {  //  HEIGHT
      for (int i = 0; i < kernel_size; i++) {  //  WIDTH
        int ji_off = ((j * kernel_size) + i) * bottomchannels;
        for (int ch = ch_off; ch < bottomchannels; ch += (THREADS_PER_WARP * WARPS_PER_BLOCK)) {
          //  CHANNELS
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;
          int idxPatchData = ji_off + ch;
          int idx2 = ((item * bottomheight + y2 + j) * bottomwidth + x2 + i) * bottomchannels + ch;
          sum[ch_off] += patch_data[idxPatchData] * bottom1[idx2];
        }
      }
    }
    __syncthreads();
    if (ch_off == 0) {
        Dtype total_sum = 0;
        for (int idx = 0; idx < THREADS_PER_WARP * WARPS_PER_BLOCK; idx++) {
            total_sum += sum[idx];
        }
        const int index = ((top_channel * topheight + blockIdx.y) * topwidth) + blockIdx.x;
        top[index + item*topcount] = total_sum;
    }  //  Aggregate result of  different threads
  }
}

//  == Correlation Backward Pass Kernel (For data1)
template <typename Dtype>
__global__ void CorrelateDataBackward0(const int nthreads, int num, int item,
  int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  Dtype *bottom0diff, const Dtype *bottom1, const Dtype *topdiff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index % bottomchannels;  //  channels
    int l = (index / bottomchannels) % bottomwidth + pad_size;  //  w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size;  //  h-pos
    //  Get X,Y ranges and clamp
    //  round_off is a trick to enable integer division with ceil, even for negative numbers
    //  We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    //  We add round_off before_s1 the int division and subtract round_off after it,
    //  to ensure the formula matches ceil behavior:
    int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1)\
     / stride1 + 1 - round_off;  //  ceil (l - 2*kernel_radius - max_displacement) / stride1
    int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1)\
     / stride1 + 1 - round_off;  //  ceil (l - 2*kernel_radius - max_displacement) / stride1
    //  Same here:
    int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off;
    //  floor (l - max_displacement) / stride1
    int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off;
    //  floor (m - max_displacement) / stride1
    Dtype sum = 0;
    if (xmax >= 0 && ymax >= 0 && (xmin <= topwidth-1) && (ymin <= topheight-1)) {
        xmin = max(0, xmin);
        xmax = min(topwidth-1, xmax);
        ymin = max(0, ymin);
        ymax = min(topheight-1, ymax);
        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
          for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
            //  Get bottom1 data:
            int s2o = stride2 * o;
            int s2p = stride2 * p;
            int idxbot1 = ((item * pbottomheight + (m + s2p)) * pbottomwidth + (l + s2o))\
             * bottomchannels + n;
            Dtype bot1tmp = bottom1[idxbot1];  // bottom1[l+s2o,m+s2p,n]
            //  Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * neighborhood_grid_width\
             + (o + neighborhood_grid_radius);  //  index [o,p]
            int idxopoffset = (item * topchannels + op);
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x;  //  topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot1tmp;
              }
            }
          }
        }
    }
    const int bot0index = ((n * bottomheight) + (m-pad_size)) * bottomwidth + (l-pad_size);
    bottom0diff[bot0index + item * bottomcount] = sum;
  }
}

// == Correlation Backward Pass Kernel (For Blob 1)
template <typename Dtype>
__global__ void CorrelateDataBackward1(const int nthreads,
  int num, int item, int topwidth, int topheight, int topchannels,
  int max_displacement, int neighborhood_grid_radius,
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2,
  int bottomwidth, int bottomheight, int pbottomwidth, int pbottomheight,
  int bottomchannels, int bottomcount, int pad_size,
  const Dtype *bottom0, Dtype *bottom1diff, const Dtype *topdiff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //  int l = index % bottomwidth + pad_size; //w-pos
    //  int m = (index / bottomwidth) % bottomheight + pad_size; //  h-pos
    //  int n = (index / bottomwidth / bottomheight) % bottomchannels; //  channels
    int n = index % bottomchannels;  //  channels
    int l = (index / bottomchannels) % bottomwidth + pad_size;  //  w-pos
    int m = (index / bottomchannels / bottomwidth) % bottomheight + pad_size;  //  h-pos
    //  round_off is a trick to enable integer division with ceil, even for negative numbers
    //  We use a large offset, for the inner part not to become negative.
    const int round_off = ROUND_OFF;
    const int round_off_s1 = stride1 * round_off;
    Dtype sum = 0;
    for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
      for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
        int s2o = stride2 * o;
        int s2p = stride2 * p;
        //  Get X,Y ranges and clamp
        //  We add round_off before_s1 the int division and subtract round_off after it,
        //  to ensure the formula matches ceil behavior:
        int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1)\
         / stride1 + 1 - round_off;
         // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1)\
         / stride1 + 1 - round_off;
        // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
        //  Same here:
        int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off;
        //  floor (l - max_displacement - s2o) / stride1
        int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off;
        //  floor (m - max_displacement - s2p) / stride1
        if (xmax >= 0 && ymax >= 0 && (xmin <= topwidth - 1) && (ymin <= topheight - 1)) {
            xmin = max(0, xmin);
            xmax = min(topwidth-1, xmax);
            ymin = max(0, ymin);
            ymax = min(topheight-1, ymax);
            //  Get bottom0 data:
            int idxbot0 = ((item * pbottomheight + (m - s2p)) \
            * pbottomwidth + (l - s2o)) * bottomchannels + n;
            Dtype bot0tmp = bottom0[idxbot0];  //  bottom1[l+s2o,m+s2p,n]
            //  Index offset for topdiff in following loops:
            int op = (p+neighborhood_grid_radius) * \
            neighborhood_grid_width + (o+neighborhood_grid_radius);  //  index [o,p]
            int idxOpOffset = (item * topchannels + op);
            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxtopdiff = (idxOpOffset * topheight + y)\
                 * topwidth + x;  //  topdiff[x,y,o,p]
                sum += topdiff[idxtopdiff] * bot0tmp;
              }
            }
        }
      }
    }
    const int bot1index = ((n * bottomheight) + (m - pad_size)) * bottomwidth + (l - pad_size);
    bottom1diff[bot1index + item * bottomcount] = sum;
  }
}

//  == Forward
//  == Dimension rearrangement Kernel
template <typename Dtype>
__global__ void blob_rearrange_kernel2(const Dtype* in, Dtype* out, int num,
int channels, int width, int height, int widthheight, int padding, int pwidthheight) {
    //  change shape from [batchsize,channel,y,x] to [batchsize,y,x,channel]
    int xy = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy >= widthheight )
        return;
    int ch = blockIdx.y;
    int n  = blockIdx.z;
    Dtype value = in[(n * channels + ch) * widthheight + xy];
    __syncthreads();  // TODO: do we really need to sync?
    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width + 2 * padding) + xpad;
    out[(n * pwidthheight + xypad) * channels + ch] = value;
}

int AssembleForward(
    at::Tensor output, 
    at::Tensor aff,
    at::Tensor input2, 
    at::Tensor rbot2, 
    int top_channels, 
    int top_height, 
    int top_width,
    int pad_size, 
    int max_displacement, 
    int neighborhood_grid_radius, 
    int neighborhood_grid_width,
    int kernel_radius, 
    int stride1, 
    int stride2,
    cudaStream_t stream)
{
    const int bnum = input2.size(0);
    const int bchannels = input2.size(1);
    const int bheight = input2.size(2);
    const int bwidth = input2.size(3);
    const int bwidthheight = bwidth * bheight;
    int threads_per_block = 16;
    dim3 totalBlocksRearr((bwidthheight - 1) / threads_per_block + 1, bchannels, bnum);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input2.type(), "rearrange input2", ([&] {
            blob_rearrange_kernel2<scalar_t>
                <<<totalBlocksRearr, threads_per_block, 0, stream>>>
                (input2.data<scalar_t>(), rbot2.data<scalar_t>(), 
                 bnum, bchannels, bwidth, bheight, bwidthheight, pad_size, bwidthheight);
            }));

    const int paddedheight = bheight + 2 * pad_size;
    const int paddedwidth = bwidth + 2 * pad_size;
    const int bottomcount = bchannels * bheight * bwidth;
    int botThreadCount = bottomcount;
    const int gridSize = (botThreadCount + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;


    for (int n = 0; n < bnum; n++) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input2.scalar_type(), "assemble forward", ([&] {
        CorrelateDataBackward0<scalar_t><<<gridSize, kMaxThreadsPerBlock, 0, stream>>>(
            botThreadCount,
            bnum, n, top_width, top_height, top_channels,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
            stride1, stride2,
            bwidth, bheight, paddedwidth, paddedheight, bchannels, bottomcount, pad_size,
            output.data<scalar_t>(), rbot2.data<scalar_t>(), aff.data<scalar_t>());
        }));
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in mx assemble forward: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

int AssembleBackward(
    at::Tensor grad_output, 
    at::Tensor rgrad_output,
    at::Tensor rbot2,
    at::Tensor aff,
    at::Tensor grad_aff,
    at::Tensor grad_input2, 
    int top_channels, 
    int top_height, 
    int top_width, 
    int pad_size, 
    int max_displacement, 
    int kernel_size,
    int neighborhood_grid_radius, 
    int neighborhood_grid_width,
    int kernel_radius, 
    int stride1, 
    int stride2, 
    cudaStream_t stream)
{
    const int bnum = grad_output.size(0);
    const int bchannels = grad_output.size(1);
    const int bheight = grad_output.size(2);
    const int bwidth = grad_output.size(3);
    const int bwidthheight = bwidth * bheight;
    const int topcount = top_width * top_height * top_channels;
    int threads_per_block = 16;
    dim3 totalBlocksRearr((bwidthheight - 1) / threads_per_block + 1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size) * (bheight + 2 * pad_size);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.type(), "rearrange grad_output", ([&] {
            blob_rearrange_kernel2<scalar_t>
                <<<totalBlocksRearr, threads_per_block, 0, stream>>>
                (grad_output.data<scalar_t>(), rgrad_output.data<scalar_t>(), 
                 bnum, bchannels, bwidth, bheight, bwidthheight, pad_size, pwidthheight);
            }));

    const int shared_memory_per_block = (kernel_size * kernel_size) * bchannels;

    const int paddedheight = bheight + 2 * pad_size;
    const int paddedwidth = bwidth + 2 * pad_size;
    const int bottomcount = bchannels * bheight * bwidth;
    int botThreadCount = bottomcount;
    const int gridSize = (botThreadCount + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;


    int topThreadCount = topcount;
    dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);
    dim3 totalBlocksCorr(top_width, top_height, bnum);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "assemble backward aff", ([&] {
            CorrelateData<scalar_t><<<totalBlocksCorr, threadsPerBlock,
            shared_memory_per_block * sizeof(scalar_t), stream>>>(
            topThreadCount,
            bnum, top_width, top_height, top_channels, topcount,
            max_displacement, neighborhood_grid_radius,
            neighborhood_grid_width, kernel_radius, kernel_size,
            stride1, stride2, paddedwidth, paddedheight, bchannels, 
            rgrad_output.data<scalar_t>(), rbot2.data<scalar_t>(), grad_aff.data<scalar_t>());
            }));

    for (int n = 0; n < bnum; n++) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            rbot2.scalar_type(), "assemble backward input2", ([&] {
        CorrelateDataBackward1<scalar_t><<<gridSize, kMaxThreadsPerBlock, 0, stream>>>(
            botThreadCount,
            bnum, n, top_width, top_height, top_channels,
            max_displacement, neighborhood_grid_radius, neighborhood_grid_width, kernel_radius,
            stride1, stride2,
            bwidth, bheight, paddedwidth, paddedheight, bchannels, bottomcount, pad_size,
            rgrad_output.data<scalar_t>(), grad_input2.data<scalar_t>(), aff.data<scalar_t>());
        }));
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in mx assemble backward: %s\n", cudaGetErrorString(err));
        return 0;
    }
    return 1;
}
