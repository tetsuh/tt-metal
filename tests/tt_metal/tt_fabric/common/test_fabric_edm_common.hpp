// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/logger.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/fabric.hpp>
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "tt_metal/fabric/erisc_datamover_builder_helper.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_host_commands.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include "ttnn/cpp/ttnn/operations/experimental/reshape/view.hpp"
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tile.hpp>

#include "umd/device/types/arch.h"
#include "umd/device/types/cluster_descriptor_types.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_set>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

enum TwoInputReaderKernelWriteMode { LOCAL_WRITEBACK, FABRIC_UNICAST, FABRIC_MULTICAST };

static constexpr size_t TEST_WORKERS_SUBDEVICE_INDEX = 0;
static constexpr size_t TEST_EDM_FABRIC_SUBDEVICE_INDEX = 1;

using subdevice_managers_t = std::unordered_map<chip_id_t, SubDeviceManagerId>;
struct SubdeviceInfo {
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_managers;
    std::unordered_map<chip_id_t, SubDeviceId> worker_subdevice_id;
    std::unordered_map<chip_id_t, SubDeviceId> fabric_subdevice_id;
};

using tt::tt_metal::distributed::MeshContainer;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshDeviceView;
using tt::tt_metal::distributed::MeshShape;
using tt::tt_metal::distributed::SystemMesh;

class BaseFabricFixture {
protected:
    tt::ARCH arch_;
    std::size_t num_devices_;
    bool device_open = false;

    // Common constants for both fixtures
    static constexpr size_t TG_NUM_DEVICES = 36;
    static constexpr size_t GALAXY_6U_NUM_DEVICES = 32;

    // Gets the appropriate mesh shape based on device configuration
    MeshShape GetDeterminedMeshShape() const {
        if (num_devices_ == TG_NUM_DEVICES || num_devices_ == GALAXY_6U_NUM_DEVICES) {
            return MeshShape{8, 4};
        } else {
            return MeshShape{2, 4};
        }
    }

    // Validates environment and hardware for tests
    void ValidateEnvironment() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            TT_THROW("This suite can only be run without TT_METAL_SLOW_DISPATCH_MODE set");
        }

        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();

        if (!(arch_ == tt::ARCH::WORMHOLE_B0 && num_devices_ >= 8 &&
              (tt::tt_metal::GetNumPCIeDevices() == 4 || tt::tt_metal::GetNumPCIeDevices() == GALAXY_6U_NUM_DEVICES))) {
            TT_THROW("This suite can only be run on T3000 or TG Wormhole devices");
        }
    }

public:
    BaseFabricFixture() : device_open(false) {}

    BaseFabricFixture(tt::tt_metal::FabricConfig fabric_config) : device_open(false) {
        tt::tt_metal::detail::InitializeFabricConfig(fabric_config);
    }

    virtual ~BaseFabricFixture() { tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED); }

    virtual void SetupDevices() = 0;
    virtual void TearDown() = 0;
};

class Fabric1DFixture : public BaseFabricFixture {
public:
    std::shared_ptr<MeshDeviceView> view_;
    std::map<chip_id_t, IDevice*> physical_devices_;

    void SetupDevices() override {
        ValidateEnvironment();

        const MeshShape cluster_shape = GetDeterminedMeshShape();
        const auto& physical_device_ids = SystemMesh::instance().get_mapped_physical_device_ids(cluster_shape);
        physical_devices_ = tt::tt_metal::detail::CreateDevices(physical_device_ids);

        std::vector<IDevice*> devices = {};
        for (auto device_id : physical_device_ids) {
            devices.push_back(physical_devices_.at(device_id));
        }

        MeshContainer<IDevice*> device_container(cluster_shape, devices);
        view_ = std::make_shared<MeshDeviceView>(device_container);
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            tt::tt_metal::detail::CloseDevices(physical_devices_);
            device_open = false;
        }
    }

    Fabric1DFixture() : BaseFabricFixture() { this->SetupDevices(); }

    Fabric1DFixture(tt::tt_metal::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~Fabric1DFixture() override { TearDown(); }
};

class MeshFabric1DFixture : public BaseFabricFixture {
public:
    std::shared_ptr<MeshDevice> mesh_device_;

    void SetupDevices() override {
        ValidateEnvironment();
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(GetDeterminedMeshShape()));
        device_open = true;
    }

    void TearDown() override {
        if (device_open) {
            mesh_device_->close();
            device_open = false;
        }
    }

    MeshFabric1DFixture() : BaseFabricFixture() { this->SetupDevices(); }

    MeshFabric1DFixture(tt::tt_metal::FabricConfig fabric_config) : BaseFabricFixture(fabric_config) {
        this->SetupDevices();
    }

    ~MeshFabric1DFixture() override {
        if (device_open) {
            TearDown();
        }
    }
};

class Fabric1DLineDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DLineDeviceInitFixture() : Fabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D) {}
};

class Fabric1DRingDeviceInitFixture : public Fabric1DFixture {
public:
    Fabric1DRingDeviceInitFixture() : Fabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D_RING) {}
};

class MeshFabric1DLineDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DLineDeviceInitFixture() : MeshFabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D) {}
};

class MeshFabric1DRingDeviceInitFixture : public MeshFabric1DFixture {
public:
    MeshFabric1DRingDeviceInitFixture() : MeshFabric1DFixture(tt::tt_metal::FabricConfig::FABRIC_1D_RING) {}
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    BufferType input_buffer_type;
    BufferType output_buffer_type;
    tt::DataFormat l1_data_format;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};

enum Correctness { Correct, Incorrect };

template <typename CONTAINER_T>
Correctness run_output_check(CONTAINER_T const& inputs, CONTAINER_T output_buffer) {
    constexpr bool debug_mode = true;

    log_info(tt::LogTest, "Checking outputs");
    bool pass = true;

    std::size_t num_printed_mismatches = 0;
    for (size_t i = 0; i < inputs.size() && num_printed_mismatches < 64; i++) {
        if (output_buffer[i] != inputs[i]) {
            if (debug_mode) {
                if (pass) {
                    log_error("Output mismatch");
                }
                log_error("[{}]: expected {} got {}", i, inputs[i], output_buffer[i]);
                num_printed_mismatches++;
            }
            pass = false;
        }
    }
    if (num_printed_mismatches > 0) {
        log_error("... (remaining mismatches omitted)");
    }

    log_info(tt::LogTest, "Output check: {}", pass ? "PASS" : "FAIL");
    return pass ? Correctness::Correct : Correctness::Incorrect;
};

static SubdeviceInfo create_subdevices(const std::vector<IDevice*>& devices) {
    SubdeviceInfo subdevice_info;
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_manager_ids;
    for (auto device : devices) {
        const auto& tensix_sub_device =
            tt_metal::SubDevice(std::array{device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0})});
        const auto& eth_sub_device = tt_metal::SubDevice(
            std::array{CoreRangeSet(), device->worker_cores(HalProgrammableCoreType::ACTIVE_ETH, SubDeviceId{0})});
        subdevice_info.sub_device_managers.insert(
            {device->id(), device->create_sub_device_manager({tensix_sub_device, eth_sub_device}, 0)});
        device->load_sub_device_manager(subdevice_info.sub_device_managers.at(device->id()));
        subdevice_info.worker_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_WORKERS_SUBDEVICE_INDEX)});
        subdevice_info.fabric_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_EDM_FABRIC_SUBDEVICE_INDEX)});
        device->set_sub_device_stall_group({subdevice_info.worker_subdevice_id.at(device->id())});
    }

    return subdevice_info;
}

static SubdeviceInfo create_worker_subdevices(const std::vector<IDevice*>& devices) {
    SubdeviceInfo subdevice_info;
    std::unordered_map<chip_id_t, SubDeviceManagerId> sub_device_manager_ids;
    for (auto device : devices) {
        const auto& tensix_sub_device =
            tt_metal::SubDevice(std::array{device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0})});
        subdevice_info.sub_device_managers.insert(
            {device->id(), device->create_sub_device_manager({tensix_sub_device}, 0)});
        device->load_sub_device_manager(subdevice_info.sub_device_managers.at(device->id()));
        subdevice_info.worker_subdevice_id.insert(
            {device->id(), device->get_sub_device_ids().at(TEST_WORKERS_SUBDEVICE_INDEX)});
        device->set_sub_device_stall_group({subdevice_info.worker_subdevice_id.at(device->id())});
    }

    return subdevice_info;
}

Correctness run_output_check(
    const std::vector<uint32_t>& all_zeros,
    const std::vector<uint32_t>& inputs,
    std::shared_ptr<Buffer>& output_buffer) {
    constexpr bool debug_mode = true;
    std::vector<uint32_t> readback_data_vec(all_zeros.size(), 0);  // init to 0 data for easier debug

    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    return run_output_check(inputs, readback_data_vec);
};

void run_programs(std::vector<Program>& programs, const std::vector<IDevice*>& devices) {
    EXPECT_EQ(programs.size(), devices.size());
    const size_t num_programs = programs.size();
    try {
        for (size_t i = 0; i < num_programs; i++) {
            tt::tt_metal::detail::CompileProgram(devices.at(i), programs.at(i));
        }
    } catch (std::exception& e) {
        log_error("Failed compile: {}", e.what());
        throw e;
    }

    log_info(tt::LogTest, "Running...");

    std::vector<std::thread> threads;
    threads.reserve(num_programs);
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        for (size_t i = 0; i < num_programs; i++) {
            threads.emplace_back(std::thread([&] { tt_metal::detail::LaunchProgram(devices.at(i), programs.at(i)); }));
        }

        std::ranges::for_each(threads, [](std::thread& t) { t.join(); });
    } else {
        for (size_t i = 0; i < num_programs; i++) {
            tt_metal::EnqueueProgram(devices.at(i)->command_queue(), programs.at(i), false);
        }

        log_debug(tt::LogTest, "Calling Finish");
        for (size_t i = 0; i < num_programs; i++) {
            tt_metal::Finish(devices.at(i)->command_queue());
        }
    }
}

std::tuple<std::shared_ptr<Buffer>, std::vector<uint32_t>> build_input_buffer(
    IDevice* first_device, size_t tensor_size_bytes, const BankedConfig& test_config) {
    auto inputs = std::vector<uint32_t>(tensor_size_bytes / sizeof(uint32_t), 0);
    std::iota(inputs.begin(), inputs.end(), 0);

    // Input buffer
    auto local_input_buffer = CreateBuffer(InterleavedBufferConfig{
        first_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});
    tt_metal::detail::WriteToBuffer(local_input_buffer, inputs);
    return {local_input_buffer, inputs};
}

static void build_and_enqueue(
    const std::vector<IDevice*>& devices, std::vector<Program>& programs, bool enqueue_only = false) {
    TT_FATAL(
        devices.size() == programs.size(),
        "Number of devices must match number of programs when calling build_and_enqueue in test");
    if (!enqueue_only) {
        for (size_t i = 0; i < devices.size(); i++) {
            tt::tt_metal::detail::CompileProgram(devices[i], programs[i]);
        }
    }
    for (size_t i = 0; i < devices.size(); i++) {
        tt_metal::EnqueueProgram(devices[i]->command_queue(), programs[i], false);
    }
}

struct EthLinkHop {
    CoreCoord hop_src;
    CoreCoord hop_dest;
};

struct ChipConnection {
    std::vector<EthLinkHop> links;
};

struct unicast_send {
    size_t distance;
};
struct mcast_send {
    size_t distance;
    size_t range;
};

using mode_variant_t = std::variant<mcast_send, unicast_send>;

static constexpr size_t PACKET_HEADER_SIZE_BYTES = sizeof(tt::tt_fabric::PacketHeader);
void generate_sender_worker_kernels(
    Program& program,
    IDevice* device,
    const CoreCoord& worker_core,
    const tt::tt_fabric::SenderWorkerAdapterSpec& worker_fabric_connection,
    const mode_variant_t& mode,
    std::size_t edm_buffer_size,
    uint32_t page_plus_header_size,
    uint32_t num_pages_total,
    uint32_t num_pages_per_edm_buffer,
    uint32_t local_worker_fabric_semaphore_id,
    uint32_t local_worker_teardown_semaphore_id,
    uint32_t local_worker_last_message_semaphore_id,
    uint32_t dram_input_buffer_base_addr,
    bool src_is_dram,
    uint32_t dram_output_buffer_base_addr,
    bool dest_is_dram,
    uint32_t worker_buffer_index_semaphore_id,
    // farthest to closest
    const std::vector<tt::tt_fabric::edm_termination_info_t>& edm_termination_infos) {
    const auto& edm_noc_core = CoreCoord(worker_fabric_connection.edm_noc_x, worker_fabric_connection.edm_noc_y);
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,      //
        num_pages_total,  //
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        num_pages_per_edm_buffer};
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_input_buffer_base_addr};

    log_trace(tt::LogTest, "\tSenderReader CT Args");
    for (const auto& arg : sender_worker_reader_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderReader RT Args");
    for (const auto& arg : sender_worker_reader_runtime_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }

    std::vector<uint32_t> sender_worker_writer_compile_args{
        num_pages_per_edm_buffer,
        num_pages_total,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        worker_fabric_connection.num_buffers_per_channel,
        dest_is_dram,
        std::holds_alternative<mcast_send>(mode) ? 1 : 0};
    log_trace(tt::LogTest, "worker_fabric_connection.edm_l1_sem_addr: {}", worker_fabric_connection.edm_l1_sem_addr);
    log_trace(tt::LogTest, "worker_buffer_index_semaphore_id: {}", worker_buffer_index_semaphore_id);
    log_trace(tt::LogTest, "last_message_semaphore_address: {}", local_worker_last_message_semaphore_id);
    log_trace(
        tt::LogTest, "Sender communicating with EDM: x={}, y={}", (uint32_t)edm_noc_core.x, (uint32_t)edm_noc_core.y);
    std::vector<uint32_t> sender_worker_writer_runtime_args{
        worker_fabric_connection.edm_buffer_base_addr,
        worker_fabric_connection.edm_l1_sem_addr,
        local_worker_fabric_semaphore_id,
        local_worker_teardown_semaphore_id,
        (uint32_t)edm_noc_core.x,
        (uint32_t)edm_noc_core.y,
        worker_fabric_connection.num_buffers_per_channel,

        worker_fabric_connection.edm_connection_handshake_addr,
        worker_fabric_connection.edm_worker_location_info_addr,
        edm_buffer_size,
        dram_output_buffer_base_addr,
        local_worker_last_message_semaphore_id,
        worker_buffer_index_semaphore_id,
        worker_fabric_connection.persistent_fabric ? 1 : 0,
        worker_fabric_connection.buffer_index_semaphore_id};

    if (std::holds_alternative<mcast_send>(mode)) {
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).distance);
        sender_worker_writer_runtime_args.push_back(std::get<mcast_send>(mode).range);
    } else {
        sender_worker_writer_runtime_args.push_back(std::get<unicast_send>(mode).distance);
    }

    get_runtime_args_for_edm_termination_infos(edm_termination_infos, sender_worker_writer_runtime_args);

    uint32_t src0_cb_index = CBIndex::c_0;
    log_trace(tt::LogTest, "\tSenderWriter CT Args");
    for (const auto& arg : sender_worker_writer_compile_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }
    log_trace(tt::LogTest, "\tSenderWriter RT Args");
    for (const auto& arg : sender_worker_writer_runtime_args) {
        log_trace(tt::LogTest, "\t\t{}", arg);
    }

    // Just want a dummy DF
    tt::DataFormat df = (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 1024   ? tt::DataFormat::Bfp8
                        : (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 2048 ? tt::DataFormat::Float16
                                                                                     : tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_plus_header_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_plus_header_size);
    CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_core, cb_src0_config);
    auto sender_worker_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/fabric_erisc_datamover_sender_worker_reader.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_worker_reader_compile_args});
    auto sender_worker_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/fabric_erisc_datamover_sender_worker_sender.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = sender_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(program, sender_worker_reader_kernel, worker_core, sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(program, sender_worker_writer_kernel, worker_core, sender_worker_writer_runtime_args);
}

bool RunLoopbackTest(
    tt_metal::IDevice* sender_device,
    tt_metal::IDevice* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,
    std::vector<Program>& programs,
    tt::tt_fabric::FabricEriscDatamoverBuilder& chip_0_edm_builder,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    auto& sender_program = programs.at(0);
    std::size_t page_plus_header_size = page_size + sizeof(tt::tt_fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto local_worker_teardown_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, worker_cores.at(0), 0);

    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////

    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages_total,
        .size_bytes = tensor_size_bytes,
        .page_size_bytes = page_size,
        .input_buffer_type = src_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    auto [local_input_buffer, inputs] = build_input_buffer(sender_device, tensor_size_bytes, test_config);

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    auto local_output_buffer = CreateBuffer(InterleavedBufferConfig{
        sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});

    tt_metal::detail::WriteToBuffer(local_output_buffer, all_zeros);

    auto local_input_buffer_address = local_input_buffer->address();
    auto local_output_buffer_address = local_output_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    // EDM Builder Setup
    ////////////////////////////////////////////////////////////////////////////

    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes + PACKET_HEADER_SIZE_BYTES;

    auto chip0_worker_fabric_connection = chip_0_edm_builder.build_connection_to_worker_channel();
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const auto& edm_config = tt::tt_fabric::FabricEriscDatamoverConfig(edm_buffer_size);
    const std::vector<tt::tt_fabric::edm_termination_info_t>& edm_termination_infos =
        enable_persistent_fabric ? std::vector<tt::tt_fabric::edm_termination_info_t>{}
                                 : std::vector<tt::tt_fabric::edm_termination_info_t>{
                                       {1,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_receiver_core).y,
                                        chip_0_edm_builder.config.termination_signal_address},
                                       {0,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).x,
                                        sender_device->ethernet_core_from_logical_core(eth_sender_core).y,
                                        chip_0_edm_builder.config.termination_signal_address}};

    TT_ASSERT(
        (enable_persistent_fabric && edm_termination_infos.size() == 0) ||
        (!enable_persistent_fabric && edm_termination_infos.size() > 0));
    generate_sender_worker_kernels(
        sender_program,
        sender_device,
        worker_core,
        chip0_worker_fabric_connection,
        unicast_send{2},  // 2 hops because we are looping back to ourselves
        edm_buffer_size,
        page_plus_header_size,
        num_pages_total,
        pages_per_send,
        local_worker_fabric_semaphore_id,
        local_worker_teardown_semaphore_id,
        local_worker_last_message_semaphore_id,
        local_input_buffer_address,
        src_is_dram,
        local_output_buffer_address,
        dest_is_dram,
        worker_buffer_index_semaphore_id,
        edm_termination_infos);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    std::vector<IDevice*> devices = {sender_device};
    if (!enable_persistent_fabric) {
        devices.push_back(receiver_device);
    }
    log_trace(tt::LogTest, "{} programs, {} devices", programs.size(), devices.size());
    run_programs(programs, devices);
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        pass &= run_output_check(all_zeros, inputs, local_output_buffer) == Correctness::Correct;
    }
    return pass;
}

void generate_multi_input_test_worker_reader_kernel(
    Program& program,
    const std::vector<uint32_t>& cb_indices,
    const std::vector<const Tensor*>& tensors,
    IDevice* device,
    uint32_t page_size,
    const CoreRangeSet& worker_core_range,
    uint32_t num_pages_per_edm_buffer,
    const ttnn::ccl::v2::TensorSlice& in0_command_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_command_tensor_slice,
    ttnn::ccl::cmd::CclCommandCode command_type,
    const DataMovementConfig& datamovement_kernel_config,
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_forward_fabric_connection,
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_backward_fabric_connection,
    const std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence>& optional_teardown_sequence,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args) {
    bool fabric_enabled = std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args) ||
                          std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args);
    using namespace ttnn::ccl::cmd::uops;
    using namespace ttnn::ccl::cmd;
    log_trace(
        tt::LogTest,
        "Generating multi input test worker reader kernel for command type: {}",
        static_cast<uint32_t>(command_type));

    TT_FATAL(
        command_type == ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB ||
            command_type == ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR,
        "Unsupported tensor IO command type");

    TT_ASSERT(tensors.size() > 0 && tensors.size() <= 2);
    TT_ASSERT(cb_indices.size() == tensors.size());

    auto sender_worker_reader_kernel = ttnn::ccl::worker_detail::generate_multi_command_stream_kernel_ct_args(
        program, cb_indices, tensors, worker_core_range, datamovement_kernel_config);

    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> ccl_command_stream0;
    std::vector<ttnn::ccl::cmd::CclHostLowLevelWorkerCommand> ccl_command_stream1;

    // Add the main tensor slice commands
    if (command_type == ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB) {
        log_trace(tt::LogTest, "Adding local noc read");
        if (fabric_enabled) {
            ccl_command_stream0.push_back(
                read_tensor_slice_to_cb_for_eventual_fabric_write(in0_command_tensor_slice, cb_indices.at(0)));
            ccl_command_stream1.push_back(
                read_tensor_slice_to_cb_for_eventual_fabric_write(in1_command_tensor_slice, cb_indices.at(1)));
        } else {
            ccl_command_stream0.push_back(read_tensor_slice_to_cb(in0_command_tensor_slice, cb_indices.at(0)));
            ccl_command_stream1.push_back(read_tensor_slice_to_cb(in1_command_tensor_slice, cb_indices.at(1)));
        }
    } else {
        if (std::holds_alternative<ttnn::ccl::cmd::LocalOnlyCommandDestArgs>(dest_args)) {
            log_trace(tt::LogTest, "Adding local noc write");
            ccl_command_stream0.push_back(local_write_cb_to_tensor_slice(in0_command_tensor_slice, cb_indices.at(0)));
            ccl_command_stream1.push_back(local_write_cb_to_tensor_slice(in1_command_tensor_slice, cb_indices.at(1)));
        } else {
            if (std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args)) {
                log_trace(
                    tt::LogTest,
                    "Adding fabric unicast write command. Distance: {}. Forward: {}",
                    std::get<UnicastCommandDestArgs>(dest_args).distance_in_hops,
                    std::get<UnicastCommandDestArgs>(dest_args).is_forward_direction);
                ccl_command_stream0.push_back(fabric_write_cb_to_tensor_slice(
                    in0_command_tensor_slice,
                    cb_indices.at(0),
                    UnicastCommandDestArgs{std::get<UnicastCommandDestArgs>(dest_args)}));
                ccl_command_stream1.push_back(fabric_write_cb_to_tensor_slice(
                    in1_command_tensor_slice,
                    cb_indices.at(1),
                    UnicastCommandDestArgs{std::get<UnicastCommandDestArgs>(dest_args)}));
            } else if (std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args)) {
                log_trace(
                    tt::LogTest,
                    "Adding fabric multicast write command. Forward: {}. Backward: {}",
                    std::get<MulticastCommandDestArgs>(dest_args).num_targets_forward_direction,
                    std::get<MulticastCommandDestArgs>(dest_args).num_targets_backward_direction);
                ccl_command_stream0.push_back(fabric_write_cb_to_tensor_slice(
                    in0_command_tensor_slice,
                    cb_indices.at(0),
                    MulticastCommandDestArgs{std::get<MulticastCommandDestArgs>(dest_args)}));
                ccl_command_stream1.push_back(fabric_write_cb_to_tensor_slice(
                    in1_command_tensor_slice,
                    cb_indices.at(1),
                    MulticastCommandDestArgs{std::get<MulticastCommandDestArgs>(dest_args)}));
            } else {
                log_trace(tt::LogTest, "WTF? Should have been caught earlier");
                TT_FATAL(true, "Unsupported dest args type");
            }
        }
    }

    // Now, because we are bringing up/tearing down the fabric per op with this program, we need to queue up the
    // commands to teardown the fabric
    // We need to make sure only one of the command streams is sending out the termination signals, and we
    // need to make sure it only does that after the other command stream is done - so what we do is
    // make the termination command stream wait for a semaphore value (locally) that the other command stream
    // will set after it has finished.
    if (optional_teardown_sequence.has_value()) {
        std::ranges::copy(optional_teardown_sequence.value(), std::back_inserter(ccl_command_stream0));
    }

    ttnn::ccl::worker_detail::generate_multi_input_command_stream_kernel_rt_args(
        program,
        sender_worker_reader_kernel,
        tensors,
        {page_size, page_size},
        device,
        num_pages_per_edm_buffer,  // TODO: get from fabric
        worker_core_range,
        ccl_command_stream0,
        ccl_command_stream1,
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection);
}

void generate_multi_input_test_worker_kernels_for_local_tensor_write(
    Program& program,
    IDevice* device,
    Tensor& input_tensor0,
    Tensor& input_tensor1,
    Tensor& output_tensor0,
    Tensor& output_tensor1,
    size_t first_cb_index,
    size_t second_cb_index,
    const CoreCoord& worker_core,
    const uint32_t page_plus_header_size,
    const uint32_t num_pages_per_edm_buffer,
    const ttnn::ccl::v2::TensorSlice& in0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out1_tensor_slice,
    const std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence>& optional_teardown_sequence,
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_forward_fabric_connection,
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& chip0_worker_backward_fabric_connection,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args) {
    // Just want a dummy DF
    tt::DataFormat df = (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 1024   ? tt::DataFormat::Bfp8
                        : (page_plus_header_size - PACKET_HEADER_SIZE_BYTES) == 2048 ? tt::DataFormat::Float16
                                                                                     : tt::DataFormat::Float32;

    {
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_plus_header_size, {{first_cb_index, df}})
                .set_page_size(first_cb_index, page_plus_header_size);
        CBHandle cb0 = CreateCircularBuffer(program, worker_core, cb_src0_config);
    }
    {
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                2 * num_pages_per_edm_buffer * page_plus_header_size, {{second_cb_index, df}})
                .set_page_size(second_cb_index, page_plus_header_size);
        CBHandle cb1 = CreateCircularBuffer(program, worker_core, cb_src1_config);
    }

    generate_multi_input_test_worker_reader_kernel(
        program,
        {first_cb_index, second_cb_index},
        {&input_tensor0, &input_tensor1},
        device,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        CoreRangeSet({CoreRange(worker_core)}),
        num_pages_per_edm_buffer,
        in0_tensor_slice,
        in1_tensor_slice,
        ttnn::ccl::cmd::CclCommandCode::STREAM_TENSOR_TO_CB,
        tt_metal::ReaderDataMovementConfig{},
        std::nullopt,
        std::nullopt,
        std::nullopt,
        dest_args);

    generate_multi_input_test_worker_reader_kernel(
        program,
        {first_cb_index, second_cb_index},
        {&output_tensor0, &output_tensor1},
        device,
        page_plus_header_size - PACKET_HEADER_SIZE_BYTES,
        CoreRangeSet({CoreRange(worker_core)}),
        num_pages_per_edm_buffer,
        out0_tensor_slice,
        out1_tensor_slice,
        ttnn::ccl::cmd::CclCommandCode::STREAM_CB_TO_TENSOR,
        tt_metal::WriterDataMovementConfig{},
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection,
        optional_teardown_sequence,
        dest_args);
}

bool RunLocalTestWithMultiInputReaders(
    const std::vector<tt_metal::IDevice*>& devices,
    std::vector<Program>& programs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,

    Tensor& input_tensor0,
    Tensor& input_tensor1,
    Tensor& output_tensor0,
    Tensor& output_tensor1,
    std::vector<Tensor> input0_tensors,   // Device
    std::vector<Tensor> input1_tensors,   // Device
    std::vector<Tensor> output0_tensors,  // Device
    std::vector<Tensor> output1_tensors,  // Device

    const ttnn::ccl::v2::TensorSlice& in0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& in1_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out0_tensor_slice,
    const ttnn::ccl::v2::TensorSlice& out1_tensor_slice,

    const uint32_t page_size,
    TwoInputReaderKernelWriteMode test_mode,
    const ttnn::ccl::cmd::CclCommandDestArgs& dest_args,
    std::optional<SubdeviceInfo>& subdevice_managers,
    bool enable_persistent_fabric) {
    const bool fabric_enabled = test_mode != TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK;
    tt_metal::IDevice* device = devices.at(0);
    for (size_t i = 0; i < devices.size(); i++) {
        log_info(tt::LogTest, "Device[{}] ID: {}", i, devices.at(i)->id());
    }
    auto program_ptrs = std::vector<Program*>();
    program_ptrs.reserve(devices.size());
    std::ranges::transform(programs, std::back_inserter(program_ptrs), [](auto& p) { return &p; });

    size_t output_tensor_dest_device_index = 0;
    if (fabric_enabled) {
        if (std::holds_alternative<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args)) {
            log_info(
                tt::LogTest,
                "Unicast command dest args. Distance in hops: {}",
                std::get<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args).distance_in_hops);
            output_tensor_dest_device_index =
                std::get<ttnn::ccl::cmd::UnicastCommandDestArgs>(dest_args).distance_in_hops;
            TT_ASSERT(output_tensor_dest_device_index != 0, "Output tensor destination device index must be non-zero");
            TT_ASSERT(test_mode == TwoInputReaderKernelWriteMode::FABRIC_UNICAST);
        } else if (std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args)) {
            log_info(
                tt::LogTest,
                "Multicast command dest args. Number of targets forward direction: {}",
                std::get<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args).num_targets_forward_direction);
            output_tensor_dest_device_index =
                std::get<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args).num_targets_forward_direction;
            TT_ASSERT(output_tensor_dest_device_index != 0, "Output tensor destination device index must be non-zero");
            TT_ASSERT(test_mode == TwoInputReaderKernelWriteMode::FABRIC_MULTICAST);
        }
    } else {
        log_info(tt::LogTest, "No fabric enabled");
        TT_ASSERT(
            std::holds_alternative<ttnn::ccl::cmd::DestTypeArgsNull>(dest_args), "Local command dest args expected");
    }

    std::size_t page_plus_header_size = page_size + sizeof(tt::tt_fabric::PacketHeader);

    auto first_cb_index = tt::CB::c_in0;
    auto second_cb_index = tt::CB::c_in1;

    auto output_tensor_dest_device = devices.at(output_tensor_dest_device_index);
    TT_ASSERT(input_tensor0.get_logical_shape()[-2] != 1);

    bool is_fabric_mcast = std::holds_alternative<ttnn::ccl::cmd::MulticastCommandDestArgs>(dest_args);

    auto input_tensor0_device = input0_tensors.at(0);
    auto input_tensor1_device = input1_tensors.at(0);
    auto output_tensor0_device = output0_tensors.at(output_tensor_dest_device_index);
    auto output_tensor1_device = output1_tensors.at(output_tensor_dest_device_index);

    log_info(tt::LogTest, "input_tensor0_device->address(): {}", input_tensor0_device.buffer()->address());
    log_info(tt::LogTest, "input_tensor1_device->address(): {}", input_tensor1_device.buffer()->address());
    log_info(
        tt::LogTest,
        "output_tensor0_device->address(): {} on device {}",
        output_tensor0_device.buffer()->address(),
        output_tensor_dest_device->id());
    log_info(
        tt::LogTest,
        "output_tensor1_device->address(): {} on device {}",
        output_tensor1_device.buffer()->address(),
        output_tensor_dest_device->id());

    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    const auto& worker_core = CoreCoord(0, 0);

    const size_t num_pages_per_edm_buffer = 2;

    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> chip0_worker_forward_fabric_connection =
        fabric_enabled ? line_fabric->uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD)
                       : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>{std::nullopt};

    // always at start of line for now
    std::optional<std::vector<tt::tt_fabric::edm_termination_info_t>> edm_termination_infos =
        (!fabric_enabled || enable_persistent_fabric)
            ? std::optional<std::vector<tt::tt_fabric::edm_termination_info_t>>{std::nullopt}
            : line_fabric->generate_ordered_termination_info_farthest_to_nearest();
    std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> chip0_worker_backward_fabric_connection = std::nullopt;

    std::optional<ttnn::ccl::SyncModeSpec> sync_details;
    std::optional<CoreCoord> teardown_worker_core;
    std::optional<ttnn::ccl::cmd::CclHostLowLevelCommandSequence> teardown_command_stream;
    if (fabric_enabled && !enable_persistent_fabric) {
        teardown_worker_core = worker_core;

        sync_details = ttnn::ccl::SyncModeSpec{};
        sync_details->core = teardown_worker_core.value();
        sync_details->add_signal(tt::tt_metal::CreateSemaphore(programs.at(0), teardown_worker_core.value(), 0), 1);
        teardown_command_stream = {ttnn::ccl::cmd::uops::local_core_semaphore_inc(sync_details->sem_ids.at(0), 1)};
        TT_FATAL(edm_termination_infos.has_value(), "EDM termination infos must be set if fabric is enabled");
        ttnn::ccl::cmd::CclHostLowLevelCommandSequence teardown_commands;

        teardown_commands = ttnn::ccl::worker_detail::build_ccl_cmd_proc_teardown_commands(
            programs.at(0),
            device,
            nullptr,  // forward device - in this test, we have a single source doing all teardown
            devices.size(),
            0,
            edm_termination_infos.value(),
            sync_details.value(),
            line_fabric.value());
        std::ranges::copy(teardown_commands, std::back_inserter(teardown_command_stream.value()));
    }

    generate_multi_input_test_worker_kernels_for_local_tensor_write(
        programs.at(0),
        device,
        input_tensor0_device,
        input_tensor1_device,
        output_tensor0_device,
        output_tensor1_device,
        first_cb_index,
        second_cb_index,
        worker_core,
        page_plus_header_size,
        num_pages_per_edm_buffer,
        in0_tensor_slice,
        in1_tensor_slice,
        out0_tensor_slice,
        out1_tensor_slice,
        teardown_command_stream,
        chip0_worker_forward_fabric_connection,
        chip0_worker_backward_fabric_connection,
        dest_args);

    if (!enable_persistent_fabric) {
        log_info(tt::LogTest, "Building EDM kernels");
        line_fabric->build_kernels();
    }

    log_info(tt::LogTest, "persistent_fabric: {}", enable_persistent_fabric);
    log_info(tt::LogTest, "subdevice_managers.has_value(): {}", subdevice_managers.has_value());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    run_programs(programs, enable_persistent_fabric ? std::vector<IDevice*>{devices[0]} : devices);
    log_info(tt::LogTest, "Finished");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        log_info(tt::LogTest, "Reading back outputs");
        auto output0_cpu = output_tensor0_device.cpu(true, ttnn::DefaultQueueId);
        auto output1_cpu = output_tensor1_device.cpu(true, ttnn::DefaultQueueId);

        auto in0_tensor_copyback_cpu = input_tensor0_device.cpu(true, ttnn::DefaultQueueId);
        auto in1_tensor_copyback_cpu = input_tensor1_device.cpu(true, ttnn::DefaultQueueId);

        auto in0_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(in0_tensor_copyback_cpu);
        auto in1_tensor_copyback = tt::tt_metal::host_buffer::get_as<uint32_t>(in1_tensor_copyback_cpu);

        auto in0_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(input_tensor0);
        auto in1_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(input_tensor1);
        auto out0_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(output0_cpu);
        auto out1_tensor_data = tt::tt_metal::host_buffer::get_as<uint32_t>(output1_cpu);

        bool input0_copyback_check_passed =
            run_output_check(in0_tensor_data, in0_tensor_copyback) == Correctness::Correct;
        bool input1_copyback_check_passed =
            run_output_check(in1_tensor_data, in1_tensor_copyback) == Correctness::Correct;
        TT_FATAL(input0_copyback_check_passed, "Input 0 copyback check failed");
        TT_FATAL(input1_copyback_check_passed, "Input 1 copyback check failed");

        log_info(tt::LogTest, "Comparing outputs");
        pass &= run_output_check(in0_tensor_data, out0_tensor_data) == Correctness::Correct;
        if (pass) {
            log_info(tt::LogTest, "Output check passed for output 0");
        } else {
            log_error(tt::LogTest, "Output check failed for output 0");
        }
        pass &= run_output_check(in1_tensor_data, out1_tensor_data) == Correctness::Correct;
        if (pass) {
            log_info(tt::LogTest, "Output check passed for output 1");
        } else {
            log_error(tt::LogTest, "Output check failed for output 1");
        }
    }

    return pass;
}

// KEEP
bool RunLineFabricTest(
    std::vector<tt_metal::IDevice*> devices,
    std::vector<Program>& programs,

    const size_t mcast_first_chip,
    const size_t mcast_last_chip,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,

    std::optional<SubdeviceInfo>& subdevice_managers,
    ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    bool enable_persistent_fabric) {
    std::size_t page_plus_header_size = page_size + sizeof(tt::tt_fabric::PacketHeader);
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    static constexpr std::size_t edm_buffer_size =
        tt::tt_fabric::FabricEriscDatamoverBuilder::default_packet_payload_size_bytes + PACKET_HEADER_SIZE_BYTES;
    const size_t local_chip_id = 0;
    const size_t remote_chip_id = 1;
    auto program_ptrs = std::vector<Program*>(devices.size());
    std::transform(programs.begin(), programs.end(), program_ptrs.begin(), [](auto& program) { return &program; });

    std::vector<CoreCoord> worker_cores = {CoreCoord(0, 0)};

    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////
    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages_total,
        .size_bytes = tensor_size_bytes,
        .page_size_bytes = page_size,
        .input_buffer_type = src_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    // Input buffer
    auto [local_input_buffer, inputs] = build_input_buffer(devices[0], tensor_size_bytes, test_config);
    auto local_input_buffer_address = local_input_buffer->address();

    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    // output buffers
    TT_ASSERT(
        enable_persistent_fabric || mcast_first_chip <= mcast_last_chip,
        "mcast_first_chip must be less than or equal to mcast_last_chip");
    TT_ASSERT(
        enable_persistent_fabric || mcast_last_chip < devices.size(),
        "mcast_last_chip must be less than the number of devices");
    std::vector<std::shared_ptr<Buffer>> output_buffers;
    output_buffers.reserve(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        if (i == 0) {
            output_buffers.push_back(CreateBuffer(InterleavedBufferConfig{
                devices.at(i), test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type}));
        } else {
            output_buffers.push_back(CreateBuffer(
                InterleavedBufferConfig{
                    devices.at(i), test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type},
                output_buffers[0]->address()));
        }
        tt_metal::detail::WriteToBuffer(output_buffers.back(), all_zeros);
    }
    auto local_output_buffer_address = output_buffers[0]->address();
    bool all_same_addr = std::ranges::all_of(output_buffers, [local_output_buffer_address](const auto& buffer) {
        return buffer->address() == local_output_buffer_address;
    });
    TT_ASSERT(all_same_addr, "All output buffers must have the same address");

    ////////////////////////////////////////////////////////////////////////////
    //   Setup Semaphores and Builders
    ////////////////////////////////////////////////////////////////////////////

    auto local_worker_fabric_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto local_worker_teardown_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto local_worker_last_message_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(programs[0], worker_cores.at(0), 0);
    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_trace(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    const auto& worker_core = worker_cores.at(0);
    log_trace(tt::LogTest, "Worker {}. On Core x={},y={}", 0, worker_core.x, worker_core.y);

    const auto edm_termination_infos = enable_persistent_fabric
                                           ? std::vector<tt::tt_fabric::edm_termination_info_t>{}
                                           : line_fabric.generate_ordered_termination_info_farthest_to_nearest();

    auto chip0_worker_fabric_connection =
        line_fabric.uniquely_connect_worker(devices[0], ttnn::ccl::EdmLineFabricOpInterface::FORWARD);

    const std::size_t pages_per_send =
        (chip0_worker_fabric_connection.buffer_size_bytes - PACKET_HEADER_SIZE_BYTES) / page_size;
    generate_sender_worker_kernels(
        programs[0],
        devices[0],
        worker_core,
        chip0_worker_fabric_connection,
        mcast_send{mcast_first_chip, mcast_last_chip - mcast_first_chip + 1},
        edm_buffer_size,
        page_plus_header_size,
        num_pages_total,
        pages_per_send,
        local_worker_fabric_semaphore_id,
        local_worker_teardown_semaphore_id,
        local_worker_last_message_semaphore_id,
        local_input_buffer_address,
        src_is_dram,
        local_output_buffer_address,
        dest_is_dram,
        worker_buffer_index_semaphore_id,
        edm_termination_infos);

    ////////////////////////////////////////////////////////////////////////////
    // Build EDM Kernels
    ////////////////////////////////////////////////////////////////////////////
    if (!enable_persistent_fabric) {
        line_fabric.build_kernels();
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    run_programs(programs, devices);
    log_info(tt::LogTest, "Reading back outputs");

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        // Check all output buffers. Make sure only the buffers in the mcast range are
        // non-zero. All other buffers outside the range should be zero filled
        TT_ASSERT(
            !std::all_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x == 0; }),
            "Input buffer expected to not be all 0");
        for (size_t i = 0; i < output_buffers.size(); i++) {
            bool compare_with_input = (mcast_first_chip <= i && i <= mcast_last_chip);
            auto& golden_tensor = compare_with_input ? inputs : all_zeros;
            pass &= run_output_check(all_zeros, golden_tensor, output_buffers.at(i)) == Correctness::Correct;
        }
    }

    return pass;
}

void persistent_fabric_teardown_sequence(
    const std::vector<IDevice*>& devices,
    std::optional<SubdeviceInfo>& subdevice_managers,
    ttnn::ccl::EdmLineFabricOpInterface& line_fabric,
    tt::tt_fabric::TerminationSignal termination_mode = tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE) {
    log_info("Tearing down fabric");

    // Wait for workers to finish
    auto d0_worker_subdevice = devices[0]->get_sub_device_ids()[TEST_WORKERS_SUBDEVICE_INDEX];
    tt_metal::Finish(devices[0]->command_queue(), {subdevice_managers->worker_subdevice_id.at(devices[0]->id())});

    // Teardown the fabric
    line_fabric.teardown_from_host(termination_mode);

    // wait for fabric teardown to finish
    std::ranges::for_each(devices, [&](IDevice* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->fabric_subdevice_id.at(d->id())});
    });
}

void setup_test_with_persistent_fabric(
    const std::vector<IDevice*>& devices,
    std::vector<Program>& programs,
    std::optional<SubdeviceInfo>& subdevice_managers,
    std::optional<std::vector<Program>>& fabric_programs,
    std::vector<Program*>& fabric_program_ptrs,
    std::optional<ttnn::ccl::EdmLineFabricOpInterface>& line_fabric,
    bool enable_persistent_fabric,
    std::optional<size_t> num_links = std::nullopt,
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
    size_t switch_interval = 0,
    bool loopback_on_last_device = false,
    bool is_galaxy = false) {
    if (enable_persistent_fabric) {
        log_info(tt::LogTest, "Enabling persistent fabric");
        fabric_programs = std::vector<Program>(devices.size());
        subdevice_managers = create_subdevices(devices);
        std::transform(
            fabric_programs->begin(), fabric_programs->end(), std::back_inserter(fabric_program_ptrs), [](auto& p) {
                return &p;
            });
    } else {
        std::transform(
            programs.begin(), programs.end(), std::back_inserter(fabric_program_ptrs), [](auto& p) { return &p; });
    }

    line_fabric = ttnn::ccl::EdmLineFabricOpInterface(
        devices, fabric_program_ptrs, enable_persistent_fabric, num_links.value_or(1), false, topology, is_galaxy);
    line_fabric->set_firmware_context_switch_interval(switch_interval);
    if (loopback_on_last_device) {
        for (auto& edm_builder : line_fabric->edm_builders_backward_direction.at(devices.back()->id())) {
            log_trace(
                tt::LogTest,
                "Implementing loopback on device {} by connecting 1D fabric endpoint to itself at x={}, y={}",
                devices.back()->id(),
                edm_builder.my_noc_x,
                edm_builder.my_noc_y);
            edm_builder.connect_to_downstream_edm(edm_builder);
        }
    }

    if (enable_persistent_fabric) {
        TT_FATAL(fabric_programs.has_value(), "Fabric programs must be set if fabric is enabled");
        TT_FATAL(devices.size() == fabric_programs->size(), "Number of devices must match number of programs");

        log_info(tt::LogTest, "Building EDM kernels");
        line_fabric->build_kernels();
        build_and_enqueue(devices, *fabric_programs);
    }
}

// RESUME HERE AND IMPLEMENT MCAST TEST
int TestLineFabricEntrypoint(
    const size_t mcast_first_chip,
    const size_t mcast_last_chip,
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    bool enable_persistent_fabric) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 4) {
        log_info("This test can only be run on T3000 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    Fabric1DFixture test_fixture;
    auto view = *(test_fixture.view_);

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 0)),
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(0, 2)),
        view.get_device(MeshCoordinate(0, 3))};
    std::vector<Program> programs(enable_persistent_fabric ? 1 : devices.size());
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> line_fabric;
    setup_test_with_persistent_fabric(
        devices,
        programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        line_fabric,
        enable_persistent_fabric);

    auto launch_workers = [&](std::vector<Program>& _programs) -> bool {
        bool success = false;
        try {
            success = RunLineFabricTest(
                enable_persistent_fabric ? std::vector<IDevice*>{devices[0]} : devices,
                _programs,
                // fabric_hops,

                mcast_first_chip,
                mcast_last_chip,

                page_size,
                num_pages_total,
                src_is_dram,
                dest_is_dram,

                subdevice_managers,
                line_fabric.value(),
                enable_persistent_fabric);

        } catch (std::exception& e) {
            log_error("Caught exception: {}", e.what());
            test_fixture.TearDown();
            return false;
        }
        return success;
    };
    bool success = launch_workers(programs);

    if (enable_persistent_fabric) {
        std::vector<Program> second_run_programs(1);
        success = launch_workers(second_run_programs);
        persistent_fabric_teardown_sequence(
            devices, subdevice_managers, line_fabric.value(), tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

static void wait_for_worker_program_completion(
    const std::vector<IDevice*>& devices, const std::optional<SubdeviceInfo>& subdevice_managers = std::nullopt) {
    if (subdevice_managers) {
        std::ranges::for_each(devices, [&](IDevice* d) {
            tt_metal::Finish(d->command_queue(), {subdevice_managers->worker_subdevice_id.at(d->id())});
        });
    } else {
        std::ranges::for_each(devices, [&](IDevice* d) { tt_metal::Finish(d->command_queue(), {}); });
    }
}
