// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <unordered_map>

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"

using TestPhysicalMeshes = tt::tt_fabric::fabric_tests::TestPhysicalMeshes;
using TestFabricFixture = tt::tt_fabric::fabric_tests::TestFabricFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;

class TestContext {
public:
    void init();
    void handle_test_config();  // parse and process test config
    void open_devices(tt::tt_metal::FabricConfig fabric_config);
    void process_traffic_config(TestTrafficConfig traffic_config);
    void close_devices();

private:
    void validate_physical_chip_id(chip_id_t physical_chip_id) const;
    TestDevice& get_test_device(chip_id_t physical_chip_id);

    TestPhysicalMeshes physical_meshes_;
    TestFabricFixture fixture_;
    std::unordered_map<chip_id_t, TestDevice> test_devices_;
};

void TestContext::validate_physical_chip_id(chip_id_t physical_chip_id) const {
    if (this->test_devices_.find(physical_chip_id) == this->test_devices_.end()) {
        tt::log_fatal(tt::LogTest, "Unknown physical chip id: {}", physical_chip_id);
        throw std::runtime_error("Unexpected physical chip id");
    }
}

TestDevice& TestContext::get_test_device(chip_id_t physical_chip_id) {
    this->validate_physical_chip_id(physical_chip_id);
    return this->test_devices_[physical_chip_id];
}

void TestContext::init() {
    this->physical_meshes_.setup_physical_meshes();
    this->fixture_.setup_devices();
    this->physical_meshes_.print_meshes();
}

void TestContext::open_devices(tt::tt_metal::FabricConfig fabric_config) {
    this->fixture_.open_devices(fabric_config);

    for (const auto& chip_id : this->fixture_.get_available_chip_ids()) {
        auto* device_handle = this->fixture_.get_device_handle(chip_id);
        this->test_devices_.emplace_back(chip_id, device_handle);
    }
}

void TestContext::process_traffic_config(TestTrafficConfig traffic_config) {
    traffic_config.validate();

    const auto src_chip_id = traffic_config.src_phys_chip_id;
    this->validate_physical_chip_id(src_chip_id);

    const auto& chip_send_type = traffic_config.data_config.chip_send_type;

    uint32_t num_sender_configs = 0;  // number of independent sender configs to be generate from this
    std::vector<chip_id_t> dst_phys_chip_ids;
    std::vector<std::unordered_map<RoutingDirection, uint32_t>> hops_vector;
    if (traffic_config.dst_phys_chip_ids.has_value()) {
        dst_phys_chip_ids = traffic_config.dst_phys_chip_ids.value();
        for (const auto& dst_chip_id : dst_phys_chip_ids) {
            this->validate_physical_chip_id(chip_id);
        }

        for (const auto& dst_chip_id : dst_phys_chip_ids) {
            hops_vector.push_back(this->physical_meshes_.get_hops_to_chip(src_chip_id, dst_chip_id));
        }
    } else if (traffic_config.num_hops.has_value()) {
        // TODO: validate dst chip ids or number of hops based on the topology
        const auto& hops = traffic_config.num_hops.value();
        // get the dest chips based on chip send type. For unicast, hops are for the same dest chip
        // for mcast, capture each chip along the mcast hops
        dst_phys_chip_ids = this->physical_meshes_.get_chips_from_hops(src_chip_id, hops, chip_send_type);
        hops_vector.push_back(hops);
    }

    // chip send type should have been sanitized by now
    if (chip_send_type == ChipSendType::CHIP_UNICAST) {
        num_sender_configs = dst_phys_chip_ids.size();
    } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
        num_sender_configs = 1;
    }

    // use the hops vector to determine the number of sender traffic configs since for mcast there will be more
    // one dst phys chip ids
    for (auto idx = 0; idx < num_sender_configs; idx++) {
        TestTrafficSenderConfig traffic_sender_config;
        traffic_sender_config.data_config = traffic_config.data_config;

        const auto dst_phys_chip_id = dst_phys_chip_ids[idx];
        const auto& hops = hops_vector[idx];
    }

    // create sender config for the src device

    // create receiver config

    // allocate sender

    // allocate receiver

    // additional handling - if mcast mode then add receivers for every chip in the route

    // if bidirectional - add sender for every receiver
}

void TestContext::close_devices() { this->fixture_.close_devices(); }

// TODO: method to get random chip send type
// TODO: method to get random noc send type
// TODO: method to get random hops (based on mode - 1D/2D)
// TODO: method to get random dest chip (based on mode - 1D/2D)

void setup_fabric(
    tt::tt_fabric::fabric_tests::TestFabricSetup fabric_setup_config, std::vector<TestDevice>& test_devices) {}

void setup_traffic_config(TestTrafficDataConfig data_config, chip_id_t src_phys_chip_id);

void setup_traffic_config(TestTrafficDataConfig data_config, chip_id_t src_phys_chip_id);

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    tt::tt_fabric::fabric_tests::parse_config(input_args);

    TestContext test_context;
    test_context.init();

    test_context.open_devices(tt::tt_metal::FabricConfig::FABRIC_1D);

    // fabric setup
    // setup_fabric()

    // all-to-all mode

    // workers setup

    // launch programs

    //

    test_context.close_devices();

    return 0;
}
