#include <torch/extension.h>

#include <cmath>

int PSROIPoolForwardLauncher(
    const at::Tensor bottom_data, const float spatial_scale, 
    const int num_rois, const int height,
    const int width, const int channels, const int pooled_height, const int pooled_width,
    const at::Tensor bottom_rois, const int group_size, const int output_dim, 
    at::Tensor top_data, at::Tensor mapping_channel);


int PSROIPoolBackwardLauncher(
    const at::Tensor top_diff, const at::Tensor mapping_channel, 
    const int num_rois, const float spatial_scale, 
    const int channels, const int height, const int width, 
    const int pooled_width, const int pooled_height, const int output_dim, 
    at::Tensor bottom_diff, const at::Tensor bottom_rois);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int psroi_pooling_forward_cuda(int pooled_height, 
                               int pooled_width, 
                               float spatial_scale, 
                               int group_size, 
                               int output_dim,
                               at::Tensor features, 
                               at::Tensor rois, 
                               at::Tensor output, 
                               at::Tensor mappingchannel) {
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(mappingchannel);

	//Get # of Rois
	int num_rois = rois.size(0);
	int size_rois = rois.size(1);
    if (size_rois != 5) {
        printf("wrong roi size\n");
        return 0;
    }

	/* // Get # of batch_size */
	/* int batch_size = THCudaTensor_size(state, features, 0); */
	/* if (batch_size!=1) */
	/* 	return 0; */

	int num_channels = features.size(1);
	int data_height = features.size(2);
	int data_width = features.size(3);

	// call the gpu kernel for psroi_pooling
	PSROIPoolForwardLauncher(features, spatial_scale, num_rois, data_height, 
        data_width, num_channels, pooled_height, pooled_width, rois, 
        group_size, output_dim, output, mappingchannel);
	return 1;
}


int psroi_pooling_backward_cuda(float spatial_scale, 
                                int output_dim, 
                                at::Tensor top_grad, 
                                at::Tensor rois, 
                                at::Tensor bottom_grad, 
                                at::Tensor mappingchannel) {
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(mappingchannel);

    int pooled_height = top_grad.size(2);
    int pooled_width = top_grad.size(3);
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5) {
        printf("wrong roi size\n");
        return 0;
    }
    /* // batch size */
    /* int batch_size = THCudaTensor_size(state, bottom_grad, 0); */
    /* if (batch_size != 1) */
    /*     return 0; */

    int num_channels = bottom_grad.size(1);
    int data_height = bottom_grad.size(2);
    int data_width = bottom_grad.size(3);

    PSROIPoolBackwardLauncher(top_grad, mappingchannel, num_rois, 
        spatial_scale, num_channels, data_height, data_width, 
        pooled_width, pooled_height, output_dim, bottom_grad, 
        rois);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &psroi_pooling_forward_cuda, "PSRoi_Pooling forward (CUDA)");
  m.def("backward", &psroi_pooling_backward_cuda, "PSRoi_Pooling backward (CUDA)");
}
