// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_socket.hpp>
#include "flatbuffers/flatbuffers.h"

namespace tt::tt_metal::distributed {

enum class SocketEndpoint : uint8_t { SENDER, RECEIVER };

struct DistributedSocketMD {
    SocketConfig config;
    DeviceAddr peer_addr;
    std::vector<uint32_t> peer_mesh_ids;
    std::vector<uint32_t> peer_chip_ids;
};

// Create send/receive socket config buffers
std::shared_ptr<MeshBuffer> create_socket_config_buffer(
    const std::shared_ptr<MeshDevice>& device, const SocketConfig& config, SocketEndpoint socket_endpoint);

// Create socket data buffer on receiver
std::shared_ptr<MeshBuffer> create_socket_data_buffer(
    const std::shared_ptr<MeshDevice>& receiver, const SocketConfig& config);

// Write socket config data to allocated buffers
void write_socket_configs(
    const std::shared_ptr<MeshBuffer>& config_buffer,
    const std::shared_ptr<MeshBuffer>& peer_config_buffer,
    const std::shared_ptr<MeshBuffer>& socket_data_buffer,
    const SocketConfig& config,
    SocketEndpoint socket_endpoint);

// (De)serialization Functions (for Host to Host handshaking)
flatbuffers::FlatBufferBuilder serialize_distributed_socket_md(const DistributedSocketMD& socket_md);
DistributedSocketMD deserialize_distributed_socket_md(const std::vector<uint8_t>& data);

//  =============== Additional utility functions  ===============

// Given a MeshDevice and a logical device coordinate, determine the device's physical mesh id
uint32_t get_physical_mesh_id(MeshDevice* mesh_device, const MeshCoordinate& coord);

}  // namespace tt::tt_metal::distributed
