// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "dataflow_api.h"

#include <cstdint>
#include <array>
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

#include "debug/dprint.h"

void kernel_main() {
    constexpr size_t num_signals_to_wait_for = get_compile_time_arg_val(0);
    constexpr size_t send_termination_signals = false;
    std::array<volatile uint32_t*, num_signals_to_wait_for> sem_addrs;
    std::array<size_t, num_signals_to_wait_for> expected_sem_counts;
    std::array<size_t, num_signals_to_wait_for> current_sem_counts;


    size_t arg_idx = 0;
    for (size_t i = 0; i < num_signals_to_wait_for; ++i) {
        sem_addrs[i] = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++)); // hack, we pass in the address instead of the semaphore id
        DPRINT << "DRAIN WAITING ON SEMAPHORE ADDR " << (uint32_t)sem_addrs[i] << " on core (" << (uint32_t)my_y[0] << ", " << (uint32_t)my_x[0] << ")\n";
        expected_sem_counts[i] = get_arg_val<uint32_t>(arg_idx++);
        current_sem_counts[i] = 0;
    }

    while (true) {
        for (size_t i = 0; i < num_signals_to_wait_for; ++i) {
            if (current_sem_counts[i] >= expected_sem_counts[i]) {
                continue;
            }

            if (current_sem_counts[i] != *sem_addrs[i]) {
                DPRINT << "DRAIN GOT SEMINC @ " << (uint32_t)sem_addrs[i] << ". NOW= " << (uint32_t)*sem_addrs[i] << "\n";
                current_sem_counts[i] = *sem_addrs[i];
            }
        }

        bool all_done = true;
        for (size_t i = 0; i < num_signals_to_wait_for; ++i) {
            if (current_sem_counts[i] < expected_sem_counts[i]) {
                all_done = false;
                break;
            }
        }
        if (all_done) {
            break;
        }
    }

    DPRINT << "DRAIN DONE\n";
}
