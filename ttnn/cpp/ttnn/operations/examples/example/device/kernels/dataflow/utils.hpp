#include <cstdint>
#include <cstring>
#include "debug/dprint.h"
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

#define ALIGN_MASK 0xF
#define ALIGNED_16B 0x10
#define SRAM false

namespace {
#define ALWI inline __attribute__((always_inline))

ALWI void enhanced_write(const uint32_t& src_addr, const uint32_t& dst_addr, const uint32_t& size_bytes) {
    DeviceZoneScopedN("alignment condition check");
    const uint32_t src_offset = src_addr & ALIGN_MASK;
    const uint32_t dst_offset = dst_addr & ALIGN_MASK;
    const uint32_t size_offset = size_bytes & ALIGN_MASK;
    if (src_offset == dst_offset) {
        // check if both addresses are under same alignment condition
        // use noc
    } else if ((src_offset | src_offset | size_offset) == 0x0) {
        // check if the units are 16B aligned
        // use mover
    } else {
        // use memcpy
    }
}

struct Mover {
    const xmov_direction_t transfer_direction = xmov_direction_t::XMOV_L1_TO_L1;
    const tdma_mover_id_t mover_number = tdma_mover_id_t::TDMA_MOVER0;  // fix temporalily
    const uint op_binary = 0x40;
    const uint write_cmd = op_binary | (mover_number << 8);

    void write_field(uint addr, uint value) {
        volatile uint* buf = reinterpret_cast<volatile uint*>(addr);
        buf[0] = value;
    }

    uint read_field(uint addr) {
        volatile uint* buf = reinterpret_cast<volatile uint*>(addr);
        return buf[0];
    }

    void configure(uint32_t src_addr, uint32_t dst_addr, uint32_t buffer_size) {
        write_field(RISCV_TDMA_REG_XMOV_DST_ADDR, (dst_addr / 16));
        write_field(RISCV_TDMA_REG_XMOV_SRC_ADDR, (src_addr / 16));
        write_field(RISCV_TDMA_REG_XMOV_SIZE, (buffer_size / 16));
        write_field(RISCV_TDMA_REG_XMOV_DIRECTION, (uint)transfer_direction);
    }

    void run() { write_field(RISCV_TDMA_REG_COMMAND_ADDR, write_cmd); }

    void wait() {
        volatile uint tdma_mover_status;
        tdma_mover_status = read_field(RISCV_TDMA_REG_COMMAND_ADDR);

        do {
            tdma_mover_status = read_field(RISCV_TDMA_REG_STATUS);
        } while (
            (tdma_mover_status & (RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK | RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK)) !=
            RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK);
    }
};

ALWI void write_through_memcpy(uint32_t src_addr, uint32_t dst_addr, uint32_t data_size_bytes) {
    std::memcpy(reinterpret_cast<uint32_t*>(dst_addr), reinterpret_cast<uint32_t*>(src_addr), data_size_bytes);
}

ALWI void write_through_noc(uint64_t src_noc_addr, uint32_t dst_local_addr, uint32_t size_bytes) {
    noc_async_read(src_noc_addr, dst_local_addr, size_bytes);
    noc_async_read_barrier();
}

ALWI void print_read_tile_nc(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}

ALWI void print_write_tile_nc(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}

ALWI void print_read_tile_br(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA1({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_INPUT_CB, TSLICE_RD_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}

ALWI void print_write_tile_br(uint32_t cb_id) {
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA1({
            DPRINT << (uint)r << " --READ--cin1-- "
                   << TileSlice(cb_id, 0, sr, TSLICE_OUTPUT_CB, TSLICE_WR_PTR, true, true) << ENDL();
        });
    }
    DPRINT << ENDL();
}
}  // namespace
