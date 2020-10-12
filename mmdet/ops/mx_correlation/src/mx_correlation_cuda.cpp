#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

int CorrelationForward(
    at::Tensor output, 
    at::Tensor input1,
    at::Tensor input2, 
    at::Tensor rbot1, 
    at::Tensor rbot2, 
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

int CorrelationBackward(
    at::Tensor grad_output, 
    at::Tensor grad_input1, 
    at::Tensor grad_input2, 
    at::Tensor rbot1, 
    at::Tensor rbot2,
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
    int num, 
    int channels, 
    int height, 
    int width, 
    cudaStream_t stream);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int mx_correlation_forward_cuda(
    at::Tensor input1, at::Tensor input2, at::Tensor rbot1, at::Tensor rbot2,
    at::Tensor output,
    int pad_size, int kernel_size, int max_displacement, 
    int stride1, int stride2) 
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(rbot1);
    CHECK_INPUT(rbot2);
    CHECK_INPUT(output);
    int num = input1.size(0);
    int channels = input1.size(1);
    int paddedbottomheight = input1.size(2) + 2 * pad_size;
    int paddedbottomwidth = input1.size(3) + 2 * pad_size;
    rbot1.resize_({num, paddedbottomheight, paddedbottomwidth, channels});
    rbot2.resize_({num, paddedbottomheight, paddedbottomwidth, channels});
    rbot1.fill_(0);
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
    output.resize_({num, top_channels, top_height, top_width});
    output.fill_(0);
    int success = CorrelationForward(output, input1, input2, rbot1, rbot2, top_channels, top_height, top_width,
        pad_size, max_displacement, kernel_size, neighborhood_grid_radius, neighborhood_grid_width,
        kernel_radius, stride1, stride2, 
        at::cuda::getCurrentCUDAStream());
    if (!success) 
        AT_ERROR("CUDA call failed");
    return success;
}

int mx_correlation_backward_cuda(
    at::Tensor rbot1, at::Tensor rbot2,
    at::Tensor grad_output, at::Tensor grad_input1, at::Tensor grad_input2,
    int pad_size, int kernel_size, int max_displacement,
    int stride1, int stride2)
{
    CHECK_INPUT(rbot1);
    CHECK_INPUT(rbot2);
    CHECK_INPUT(grad_output);
    CHECK_INPUT(grad_input1);
    CHECK_INPUT(grad_input2);
    int paddedbottomheight = grad_input1.size(2) + 2 * pad_size;
    int paddedbottomwidth = grad_input1.size(3) + 2 * pad_size;
    int kernel_radius = (kernel_size - 1) / 2;
    int border_size = max_displacement + kernel_radius;
    int top_width = std::ceil(static_cast<float>(paddedbottomwidth - border_size * 2) \
            / static_cast<float>(stride1));
    int top_height = std::ceil(static_cast<float>(paddedbottomheight - border_size * 2) \
            / static_cast<float>(stride2));
    int neighborhood_grid_radius = max_displacement / stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
    int top_channels = neighborhood_grid_width * neighborhood_grid_width;
    int num = grad_input1.size(0);
    int channels = grad_input1.size(1);
    int height = grad_input1.size(2);
    int width = grad_input1.size(3);
    int success = CorrelationBackward(
        grad_output, grad_input1, grad_input2, rbot1, rbot2,
        top_channels, top_height, top_width, 
        pad_size, max_displacement, kernel_size,
        neighborhood_grid_radius, neighborhood_grid_width,
        kernel_radius, stride1, stride2, 
        num, channels, height, width, 
        at::cuda::getCurrentCUDAStream());
    if (!success)
        AT_ERROR("CUDA call failed");
    return success;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mx_correlation_forward_cuda, "Mx Correlation forward (CUDA)");
    m.def("backward", &mx_correlation_backward_cuda, "Mx Correlation backward (CUDA)");
}
