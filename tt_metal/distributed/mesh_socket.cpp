// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::distributed {

struct DistributedSocketMD {
    SocketConfig config;
    DeviceAddr peer_addr;
    std::vector<uint32_t> peer_mesh_ids;
    std::vector<uint32_t> peer_chip_ids;
};

DistributedSocketMD get_socket_metadata(const SocketConfig& config, MPI_Comm comm) {
    TT_FATAL(
        !(config.sender_device and config.receiver_device),
        "Cannot set both sender_device and receiver_device when ranks differ in SocketConfig");
    bool is_sender = config.sender_device;
    uint32_t peer_rank = is_sender ? config.receiver_rank : config.sender_rank;
    MPI_Status probe_status;
    MPI_Probe(peer_rank, 0, comm, &probe_status);
    if (config.sender_device) {
    }
}

MeshSocket MeshSocket::create_distributed_socket(const SocketConfig& config, MPI_Comm comm) {
    if (config.sender_rank == config.receiver_rank) {
        TT_FATAL(
            config.sender_device and config.receiver_device,
            "Sender and receiver mesh devices must be poulated when ranks are equal.");
        return MeshSocket::create_sockets(config.sender_device, config.receiver_device, config);
    }
    if (config.sender_device) {
        // Creating a sender socket
        auto config_buffer = create_socket_config_buffer(config.sender_device, config, SocketEndpoint::SENDER);
        auto recv_socket_md = get_socket_metadata(config, comm);
    }
}

std::pair<MeshSocket, MeshSocket> MeshSocket::create_sockets(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const SocketConfig& config) {
    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    auto sender_config_buffer = create_socket_config_buffer(sender, config, SocketEndpoint::SENDER);
    auto recv_config_buffer = create_socket_config_buffer(receiver, config, SocketEndpoint::RECEIVER);
    auto socket_data_buffer = create_socket_data_buffer(receiver, config);
    write_socket_configs(sender_config_buffer, recv_config_buffer, socket_data_buffer, config, SocketEndpoint::SENDER);
    write_socket_configs(
        recv_config_buffer, sender_config_buffer, socket_data_buffer, config, SocketEndpoint::RECEIVER);
    auto sender_socket = MeshSocket(
        nullptr,  // The sender socket does not have a data-buffer allocated
        sender_config_buffer,
        config);
    auto receiver_socket = MeshSocket(socket_data_buffer, recv_config_buffer, config);
    return {sender_socket, receiver_socket};
}

std::shared_ptr<MeshBuffer> MeshSocket::get_data_buffer() const {
    TT_FATAL(data_buffer_, "Cannot access the data buffer for a sender socket.");
    return data_buffer_;
};

std::shared_ptr<MeshBuffer> MeshSocket::get_config_buffer() const { return config_buffer_; }

const SocketConfig& MeshSocket::get_config() const { return config_; }

}  // namespace tt::tt_metal::distributed
