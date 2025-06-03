// mpi_send_recv_benchmark.cpp
// Improved MPI send/receive micro‑benchmark using Google Benchmark
// Compile with: mpicxx -O3 -std=c++20 -lbenchmark -lpthread -o mpi_bench mpi_send_recv_benchmark.cpp

#include <benchmark/benchmark.h>
#include <fmt/core.h>
#include <mpi.h>

#include <chrono>
#include <thread>
#include <vector>

#include "autograd/auto_context.hpp"

// #define _POSIX_C_SOURCE 200112L
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;

// User‑defined literals for convenient byte sizes
constexpr std::size_t operator"" _KiB(unsigned long long k) {
    return k * 1024ULL;
}
constexpr std::size_t operator"" _MiB(unsigned long long m) {
    return m * 1024_KiB;
}

/// Silent reporter used by non‑root ranks to suppress redundant console output
class SilentReporter final : public benchmark::BenchmarkReporter {
public:
    bool ReportContext(const Context &) noexcept override {
        return true;
    }
    void ReportRuns(const std::vector<Run> &) noexcept override {
    }
    void Finalize() noexcept override {
    }
};

#define SERVER_IP "11.228.0.10"
#define PORT 4446
#define BUF_SZ 16384
#define CHUNK_UDP 1400
#define ONE_MB (1024UL * 1024UL)

static void fill_random(uint8_t *dst, size_t len) {
    // int fd = open("/dev/urandom", O_RDONLY);
    // if (fd >= 0) {
    //     ssize_t n = read(fd, dst, len);
    //     if ((size_t)n == len) {
    //         close(fd);
    //         return;
    //     }
    //     close(fd);
    // }
    /* Fallback to rand() */
    for (size_t i = 0; i < len; ++i) dst[i] = i % 256;
}

template <std::size_t Bytes>
class IUDPSendRecv {
public:
    virtual ~IUDPSendRecv() = default;
    virtual void initialize() = 0;
    virtual void recv() {
        throw std::runtime_error("recv() not implemented");
    }
    virtual void send() {
        throw std::runtime_error("send() not implemented");
    }
};

template <std::size_t Bytes>
class UDPServer : public IUDPSendRecv<Bytes> {
public:
    ~UDPServer() override {
        close(s);
    }

    // void initialize() override {
    //     s = socket(AF_INET, SOCK_DGRAM, 0);
    //     if (s < 0) {
    //         throw std::runtime_error("Failed to create UDP socket");
    //     }

    //     addr = {.sin_family = AF_INET, .sin_addr = {htonl(INADDR_ANY)}, .sin_port = htons(PORT)};
    //     if (bind(s, (struct sockaddr *)&addr, sizeof addr) < 0) {
    //         perror("bind");
    //         close(s);
    //         throw std::runtime_error("Failed to bind UDP socket");
    //     }

    //     timeval tv{.tv_sec = 1, .tv_usec = 0};
    //     setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    // }

    void initialize() override {
        s = socket(AF_INET, SOCK_DGRAM, 0);
        if (s < 0)
            throw std::runtime_error("socket");

        /* allow rebinding and enlarge rcvbuf so we don't drop packets */
        int yes = 1, big = 4 * 1024 * 1024;  // 4 MiB
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
        setsockopt(s, SOL_SOCKET, SO_RCVBUF, &big, sizeof big);

        addr = {.sin_family = AF_INET, .sin_addr = {htonl(INADDR_ANY)}, .sin_port = htons(PORT)};
        if (bind(s, (sockaddr *)&addr, sizeof addr) < 0)
            perror("bind"), abort();

        /* keep the 1-second timeout */
        timeval tv{.tv_sec = 1, .tv_usec = 0};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    }

    void recv() override {
        struct sockaddr_in peer;
        socklen_t len = sizeof peer;

        uint64_t total_received = 0;
        while (total_received < Bytes) {
            size_t remaining = Bytes - total_received;
            ssize_t n = recvfrom(
                s,
                full_buf + total_received,
                remaining < CHUNK_UDP ? remaining : CHUNK_UDP,
                0,
                (struct sockaddr *)&peer,
                &len);
            if (n < 0) {
                perror("recvfrom");
                close(s);
                throw std::runtime_error("Failed to receive data on UDP socket");
            }
            total_received += n;
            // fmt::println(
            //     "[{}/{}] Received {} bytes from {}:{}",
            //     total_received,
            //     Bytes,
            //     n,
            //     inet_ntoa(peer.sin_addr),
            //     ntohs(peer.sin_port));
        }

        // fmt::println("Total received bytes: {}", total_received);
        // fmt::println("Checking data integrity...");

        // for (int i = 0; i < Bytes; ++i) {
        //     if (full_buf[i] != (i % 256)) {
        //         fprintf(stderr, "Data mismatch at byte %d: expected %d, got %d\n", i, i % 256, full_buf[i]);
        //         close(s);
        //         throw std::runtime_error("Data mismatch in received UDP packet");
        //     }
        // }
        // fmt::println("Received {} bytes in total", total_received);
        // fmt::println("Data integrity check passed");
    }

private:
    int s;
    sockaddr_in addr;
    uint8_t buf[CHUNK_UDP];
    uint8_t full_buf[Bytes];
};

template <std::size_t Bytes>
class UDPClient : public IUDPSendRecv<Bytes> {
public:
    ~UDPClient() override {
        close(s);
    };

    void initialize() override {
        s = socket(AF_INET, SOCK_DGRAM, 0);
        if (s < 0) {
            perror("socket");
            throw std::runtime_error("Failed to create UDP socket");
        }

        dst = {.sin_family = AF_INET, .sin_port = htons(PORT)};
        if (inet_pton(AF_INET, SERVER_IP, &dst.sin_addr) != 1) {
            perror("inet_pton");
            close(s);
            throw std::runtime_error("Failed to convert IP address");
        }

        fill_random(buf, Bytes);
    }

    void send() override {
        size_t off = 0;
        socklen_t len = sizeof dst;
        while (off < Bytes) {
            size_t chunk = (Bytes - off > CHUNK_UDP) ? CHUNK_UDP : Bytes - off;
            ssize_t n = sendto(s, buf + off, chunk, 0, (struct sockaddr *)&dst, sizeof dst);
            if (n < 0) {
                perror("sendto");
                close(s);
                throw std::runtime_error("Failed to send data on UDP socket");
            }
            off += n;
            // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    int s;
    sockaddr_in dst;
    uint8_t buf[Bytes];
    uint8_t echo[CHUNK_UDP];
};

template <std::size_t Bytes>
static void SendRecvUDP(benchmark::State &state) {
    auto &dist_ctx = ttml::autograd::ctx().get_distributed_context();
    const int world_rank = *dist_ctx.rank();

    // Synchronise ranks before measurement starts

    IUDPSendRecv<Bytes> *udp = nullptr;
    if (world_rank == 0) {
        udp = new UDPServer<Bytes>();
        udp->initialize();
    } else if (world_rank == 1) {
        udp = new UDPClient<Bytes>();
        udp->initialize();
    }
    dist_ctx.barrier();

    std::vector<std::byte> buffer(Bytes);

    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();

        if (world_rank == 0) {
            udp->recv();
        } else if (world_rank == 1) {
            udp->send();
        }

        const double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // Gather the slowest time to ensure consistent statistics
        double max_sec{};
        MPI_Allreduce(&elapsed_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_sec);
    }

    state.SetBytesProcessed(static_cast<int64_t>(Bytes) * state.iterations());

    dist_ctx.barrier();

    delete udp;
}

//-----------------------------------------------------------------------------//
// Core benchmark: round‑trip ping–pong between rank 0 and rank 1
//-----------------------------------------------------------------------------//

template <std::size_t Bytes>
static void SendRecv(benchmark::State &state) {
    auto &dist_ctx = ttml::autograd::ctx().get_distributed_context();
    const int world_rank = *dist_ctx.rank();

    // Synchronise ranks before measurement starts
    dist_ctx.barrier();

    std::vector<std::byte> buffer(Bytes);

    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();

        if (world_rank == 0) {
            dist_ctx.send(buffer, Rank{1}, Tag{0});
        } else if (world_rank == 1) {
            dist_ctx.recv(buffer, Rank{0}, Tag{0});
        }

        const double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // Gather the slowest time to ensure consistent statistics
        double max_sec{};
        MPI_Allreduce(&elapsed_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_sec);
    }

    state.SetBytesProcessed(static_cast<int64_t>(Bytes) * state.iterations());
}

template <std::size_t Bytes>
static void Broadcast(benchmark::State &state) {
    auto &dist_ctx = ttml::autograd::ctx().get_distributed_context();
    const int world_rank = *dist_ctx.rank();

    // Synchronise ranks before measurement starts
    dist_ctx.barrier();

    int broadcaster_rank = 0;

    std::vector<std::byte> buffer(Bytes);

    for (auto _ : state) {
        const auto t0 = std::chrono::steady_clock::now();

        dist_ctx.broadcast(buffer, Rank{broadcaster_rank});

        const double elapsed_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        // Gather the slowest time to ensure consistent statistics
        double max_sec{};
        MPI_Allreduce(&elapsed_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(max_sec);
    }

    state.SetBytesProcessed(static_cast<int64_t>(Bytes) * state.iterations());
}

// Register benchmark instances
BENCHMARK_TEMPLATE(SendRecv, 1_MiB)->UseManualTime()->Iterations(100);
BENCHMARK_TEMPLATE(SendRecv, 25_MiB)->UseManualTime()->Iterations(100);

BENCHMARK_TEMPLATE(Broadcast, 1_MiB)->UseManualTime()->Iterations(100);
BENCHMARK_TEMPLATE(Broadcast, 25_MiB)->UseManualTime()->Iterations(100);

BENCHMARK_TEMPLATE(SendRecvUDP, 1_MiB)->UseManualTime()->Iterations(100);
BENCHMARK_TEMPLATE(SendRecvUDP, 25_MiB)->UseManualTime()->Iterations(100);

}  // namespace

int main(int argc, char *argv[]) {
    auto &ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    auto &dist_ctx = ctx.get_distributed_context();

    // Abort early if the world is too small
    if (*dist_ctx.size() < 2) {
        if (*dist_ctx.rank() == 0)
            fmt::print(stderr, "[warning] Need at least 2 MPI ranks to run benchmark.\n");
        return 0;
    }

    benchmark::Initialize(&argc, argv);

    const bool is_root = (*dist_ctx.rank() == 0);
    if (is_root) {
        benchmark::RunSpecifiedBenchmarks();
    } else {
        SilentReporter silent;
        benchmark::RunSpecifiedBenchmarks(&silent);
    }

    return 0;
}
