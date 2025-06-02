#include <gtest/gtest.h>
#include <mpi.h>
#include <tt-metalium/distributed.hpp>
#include "tests/tt_metal/distributed/utils.hpp"

class MPITest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    int rank;
    int size;
};

struct ConfigBase {
    uint32_t data[32];
};

struct MemAttr {
    uint16_t data[47];
};

class Config {
public:
    std::vector<ConfigBase> base_configs;
    MemAttr mem_attr;

    int calculate_pack_size(MPI_Comm comm) const {
        int total_size = 0;
        int temp_size;

        // Size for vector count
        MPI_Pack_size(1, MPI_UINT32_T, comm, &temp_size);
        total_size += temp_size;

        // Size for vector data (each ConfigBase has 32 uint32_t elements)
        if (!base_configs.empty()) {
            MPI_Pack_size(base_configs.size() * 32, MPI_UINT32_T, comm, &temp_size);
            total_size += temp_size;
        }

        // Size for MemAttr (47 uint16_t elements)
        MPI_Pack_size(47, MPI_UINT16_T, comm, &temp_size);
        total_size += temp_size;

        return total_size;
    }

    int pack_data(void* buffer, int buffer_size, MPI_Comm comm) {
        int position = 0;
        uint32_t vec_size = static_cast<uint32_t>(base_configs.size());
        MPI_Pack(&vec_size, 1, MPI_UINT32_T, buffer, buffer_size, &position, comm);
        for (const auto& cfg : base_configs) {
            MPI_Pack(cfg.data, 32, MPI_UINT32_T, buffer, buffer_size, &position, comm);
        }
        MPI_Pack(mem_attr.data, 47, MPI_UINT16_T, buffer, buffer_size, &position, comm);
        return position;
    }

    int unpack_data(const void* buffer, int buffer_size, MPI_Comm comm) {
        int position = 0;
        uint32_t vec_size;
        MPI_Unpack(const_cast<void*>(buffer), buffer_size, &position, &vec_size, 1, MPI_UINT32_T, comm);
        base_configs.resize(vec_size);
        for (auto& cfg : base_configs) {
            MPI_Unpack(const_cast<void*>(buffer), buffer_size, &position, cfg.data, 32, MPI_UINT32_T, comm);
        }
        MPI_Unpack(const_cast<void*>(buffer), buffer_size, &position, mem_attr.data, 47, MPI_UINT16_T, comm);

        return position;
    }
};

TEST_F(MPITest, BasicCommunication) {
    using namespace tt::tt_metal::distributed;
    EXPECT_EQ(size, 2);  // Ensure we're running with 2 processes
    auto random_seed = 0;
    uint32_t seed = 0;  // tt::parse_env("TT_METAL_SEED", random_seed);
    if (rank == 0) {
        std::size_t var_size = 5;
        Config config;
        config.base_configs.resize(var_size);
        for (auto i = 0; i < var_size; i++) {
            auto& base_cfg = config.base_configs[i];
            for (int j = 0; j < 32; j++) {
                base_cfg.data[j] = j;
            }
        }
        for (auto i = 0; i < 47; i++) {
            config.mem_attr.data[i] = i * 2;
        }
        auto buffer_size = config.calculate_pack_size(MPI_COMM_WORLD);
        std::cout << "Buffer Size: " << buffer_size << std::endl;
        char* buffer = new char[buffer_size];
        int packed_size = config.pack_data(buffer, buffer_size, MPI_COMM_WORLD);
        std::cout << "Packed Size: " << packed_size << std::endl;
        MPI_Send(buffer, packed_size, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Status probe_status;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &probe_status);

        int buffer_size;
        MPI_Get_count(&probe_status, MPI_PACKED, &buffer_size);
        std::cout << "Recv has buffer of size: " << buffer_size << std::endl;
        char* buffer = new char[buffer_size];
        int result = MPI_Recv(buffer, buffer_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        Config config;
        config.unpack_data(buffer, buffer_size, MPI_COMM_WORLD);
        for (auto i = 0; i < config.base_configs.size(); i++) {
            std::cout << "Print base config: " << i << std::endl;
            const auto& base_cfg = config.base_configs[i];
            for (int j = 0; j < 32; j++) {
                std::cout << base_cfg.data[j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Print Mem Attr:" << std::endl;
        for (auto i = 0; i < 47; i++) {
            std::cout << config.mem_attr.data[i] << " ";
        }
        std::cout << std::endl;
    }
}

// Custom main function to initialize MPI
int main(int argc, char** argv) {
    // Initialize MPI first
    MPI_Init(&argc, &argv);

    // Get rank to control output
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Only rank 0 prints test results to avoid duplicate output
    if (rank != 0) {
        ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
