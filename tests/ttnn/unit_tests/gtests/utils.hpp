

namespace ttnn {
namespace operations {
namespace conv::conv2d {
namespace test {

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
    const std::array<uint32_t, 2>& padding);

}  // namespace test
}  // namespace conv::conv2d
}  // namespace operations
}  // namespace ttnn
