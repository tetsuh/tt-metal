#include "utils.hpp"

namespace ttnn {
namespace operations {
namespace conv::conv2d {
namespace test {

/*
    Reference implementation of Conv2D

    Takes in input tensor with original shape (N,Ci,H,W) that is flattened in row major order

    and flattened kernel tensor with original shape (Co,Ci,KH,KW) that is also flattened in row major order.

    Returns flattened tensor with original shape (N,Co,Xh,Xw) in row major order, where Xh and Xw are calculated based
    on input tensor,kernel tensor, stride and padding.


    The output vector is flattened in row major order.

    input_channels - Ci
    output_channels - Co
    input_height - H
    input_width - W
    batch_size - N
    output_height - Xh
    output_width - Xw
    kernel_size - (KH,KW)
    stride - (SH,SW)
    padding - (PH,PW)
*/
std::vector<float> reference_implementation_conv2d(
    const std::vector<float>& input,   // (N,Ci,H,W)
    const std::vector<float>& kernel,  // (Co,Ci,H',W')
    const uint32_t input_channels,
    const uint32_t output_channels,
    const uint32_t batch_size,
    const uint32_t input_height,
    const uint32_t input_width,
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding) {
    uint32_t kernel_height = kernel_size[0];
    uint32_t kernel_width = kernel_size[1];
    uint32_t padding_height = padding[0];
    uint32_t padding_width = padding[1];
    uint32_t stride_height = stride[0];
    uint32_t stride_width = stride[1];

    // Calculate output height and width
    uint32_t Xh = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    uint32_t Xw = (input_width - kernel_width + 2 * padding_width) / stride_width + 1;

    std::vector<float> output(batch_size * output_channels * Xh * Xw);
    uint32_t i = 0;
    for (uint32_t n = 0; n < batch_size; n++) {
        for (uint32_t co = 0; co < output_channels; co++) {
            for (uint32_t oh = 0; oh < Xh; oh++) {
                for (uint32_t ow = 0; ow < Xw; ow++) {
                    int32_t h = oh * stride_height - padding_height;
                    int32_t w = ow * stride_width - padding_width;
                    float sum = 0;
                    for (uint32_t ci = 0; ci < input_channels; ci++) {
                        for (uint32_t kh = 0; kh < kernel_height; kh++) {
                            for (uint32_t kw = 0; kw < kernel_width; kw++) {
                                int32_t ih = h + kh;
                                int32_t iw = w + kw;
                                if (ih >= 0 && ih < (int32_t)input_height && iw >= 0 && iw < (int32_t)input_width) {
                                    sum += input
                                               [n * input_channels * input_height * input_width +
                                                ci * input_height * input_width + ih * input_width + iw] *
                                           kernel
                                               [co * input_channels * kernel_height * kernel_width +
                                                ci * kernel_height * kernel_width + kh * kernel_width + kw];
                                }
                            }
                        }
                    }
                    output[i++] = sum;
                }
            }
        }
    }
    return output;
}

}  // namespace test
}  // namespace conv::conv2d
}  // namespace operations
}  // namespace ttnn
