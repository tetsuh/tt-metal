// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_profiler.hpp"

#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

#if defined(TRACY_ENABLE)
constexpr bool is_tracy_enabled = true;
#else
constexpr bool is_tracy_enabled = false;
#endif

namespace ttml::core {
void TTProfiler::dump_results(ttnn::distributed::MeshDevice* device, tt::tt_metal::ProfilerDumpState dump_state) const {
    assert(device);
    if (!m_enabled) {
        return;
    }
    call_device_noop(device, 2);
    for (auto& dev : device->get_devices()) {
        tt::tt_metal::detail::DumpDeviceProfileResults(dev, dump_state);
    }
}
void TTProfiler::call_device_noop(ttnn::distributed::MeshDevice* device, int count) const {
    assert(device);
    if (!m_enabled) {
        return;
    }
    auto fake_tensor = ttml::core::from_vector({1.F}, ttml::core::create_shape({1, 1, 1, 1}), device);
    for (int i = 0; i < count; ++i) {
        [[maybe_unused]] auto _ = ttml::metal::profiler_no_op(fake_tensor);
    }
}

bool TTProfiler::is_enabled() const {
    return m_enabled;
}
void TTProfiler::enable() {
    m_enabled = true;
}
void TTProfiler::disable() {
    m_enabled = false;
}
TTProfiler::TTProfiler() {
    if (is_tracy_enabled) {
        enable();
    }
}
}  // namespace ttml::core
