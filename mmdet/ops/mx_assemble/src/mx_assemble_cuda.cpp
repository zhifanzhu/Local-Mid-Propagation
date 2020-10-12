#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

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
    cudaStream_t stream);

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
    cudaStream_t stream);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int mx_assemble_forward_cuda(
    at::Tensor aff, at::Tensor input2, at::Tensor rbot2, at::Tensor output,
    int pad_size, int kernel_size, int max_displacement, 
    int stride1, int stride2) 
{
    CHECK_INPUT(aff);
    CHECK_INPUT(input2);
    CHECK_INPUT(rbot2);
    CHECK_INPUT(output);
    int num = input2.size(0);
    int channels = input2.size(1);
    int paddedbottomheight = input2.size(2) + 2 * pad_size;
    int paddedbottomwidth = input2.size(3) + 2 * pad_size;
    rbot2.resize_({num, paddedbottomheight, paddedbottomwidth, channels});
    rbot2.fill_(0);
    int kernel_radius = (kernel_size - 1) / 2;
    int border_size = max_displacement + kernel_radius;
    int top_width = std::ceil(static_cast<float>(paddedbottomwidth - border_size * 2) \
            / static_cast<float>(stride1));
    int top_height = std::ceil(static_cast<float>(paddedbottomheight - border_size * 2) \
            / static_cast<float>(stride2));
    int neighborhood_grid_radius = max_displacement / stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
    int top_channels = neighborhood_grid_width * neighborhood_grid_width;
    /* output.resize_({num, top_channels, top_height, top_width}); */
    /* output.fill_(0); */
    int success = AssembleForward(output, aff, input2, rbot2, top_channels, top_height, top_width,
        pad_size, max_displacement, neighborhood_grid_radius, neighborhood_grid_width,
        kernel_radius, stride1, stride2, 
        at::cuda::getCurrentCUDAStream());
    if (!success) 
        AT_ERROR("CUDA call failed");
    return success;
}

int mx_assemble_backward_cuda(
    at::Tensor grad_output, at::Tensor rgrad_output, at::Tensor rbot2,
    at::Tensor aff, at::Tensor grad_aff, at::Tensor grad_input2,
    int pad_size, int kernel_size, int max_displacement,
    int stride1, int stride2)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(rgrad_output);
    CHECK_INPUT(rbot2);
    CHECK_INPUT(aff);
    CHECK_INPUT(grad_aff);
    CHECK_INPUT(grad_input2);
    int paddedbottomheight = grad_output.size(2) + 2 * pad_size;
    int paddedbottomwidth = grad_output.size(3) + 2 * pad_size;
    int kernel_radius = (kernel_size - 1) / 2;
    int border_size = max_displacement + kernel_radius;
    int top_width = std::ceil(static_cast<float>(paddedbottomwidth - border_size * 2) \
            / static_cast<float>(stride1));
    int top_height = std::ceil(static_cast<float>(paddedbottomheight - border_size * 2) \
            / static_cast<float>(stride2));
    int neighborhood_grid_radius = max_displacement / stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
    int top_channels = neighborhood_grid_width * neighborhood_grid_width;
    int num = grad_output.size(0);
    int channels = grad_output.size(1);
    rgrad_output.resize_({num, paddedbottomheight, paddedbottomwidth, channels});
    rgrad_output.fill_(0);
    int success = AssembleBackward(
        grad_output, rgrad_output, rbot2,
        aff, grad_aff, grad_input2,
        top_channels, top_height, top_width, 
        pad_size, max_displacement, kernel_size,
        neighborhood_grid_radius, neighborhood_grid_width,
        kernel_radius, stride1, stride2, 
        at::cuda::getCurrentCUDAStream());
    if (!success)
        AT_ERROR("CUDA call failed");
    return success;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mx_assemble_forward_cuda, "Mx Assemble forward (CUDA)");
    m.def("backward", &mx_assemble_backward_cuda, "Mx Assemble backward (CUDA)");
}
