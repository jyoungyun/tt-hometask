// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_binary.h"

#define REDUCE_OP   PoolType::MAX
#define REDUCE_DIM  ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    // Initialize the SFPU
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);

    fill_tile_init();

    for (uint32_t i = 0; i < n_tiles; ++i) {
        // Get Max
    reduce_init<PoolType::MAX, REDUCE_DIM>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_1,
        tt::CBIndex::c_1
    );

    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_reserve_back(tt::CBIndex::c_1, 1);

    tile_regs_acquire();
    reduce_tile<PoolType::MAX, REDUCE_DIM>(
        tt::CBIndex::c_0, tt::CBIndex::c_1,
        0, 0, 0
    );
    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, tt::CBIndex::c_1);
    tile_regs_release();
    cb_push_back(tt::CBIndex::c_1, 1);

    reduce_uninit();

    // x - max -> exp
    exp_tile_init();
    sub_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);

    cb_wait_front(tt::CBIndex::c_1, 1);  // max
    cb_reserve_back(tt::CBIndex::c_3, 1);

    tile_regs_acquire();
    sub_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, tt::CBIndex::c_3);
    tile_regs_release();

    cb_push_back(tt::CBIndex::c_3, 1);
    cb_pop_front(tt::CBIndex::c_0, 1);
    cb_pop_front(tt::CBIndex::c_1, 1);

    // Get Sum
    reduce_init<PoolType::SUM, REDUCE_DIM>(
        tt::CBIndex::c_3,
        tt::CBIndex::c_2,
        tt::CBIndex::c_2
    );

    cb_wait_front(tt::CBIndex::c_3, 1); // exp
    cb_reserve_back(tt::CBIndex::c_2, 1);

    tile_regs_acquire();
    reduce_tile<PoolType::SUM, REDUCE_DIM>(
        tt::CBIndex::c_3, tt::CBIndex::c_2,
        0, 0, 0
    );
    tile_regs_commit();
    tile_regs_wait();

    rdiv_tile_init();
    rdiv_tile(0, 1);  // 1/sum

    pack_tile(0, tt::CBIndex::c_2);
    tile_regs_release();

    cb_push_back(tt::CBIndex::c_2, 1);

    reduce_uninit();

    // exp / sum -> output
    mul_tiles_init(tt::CBIndex::c_3, tt::CBIndex::c_2);

    cb_wait_front(tt::CBIndex::c_3, 1);  // exp
    cb_wait_front(tt::CBIndex::c_2, 1);  // sum
    cb_reserve_back(tt::CBIndex::c_16, 1);

    tile_regs_acquire();
    mul_tiles(tt::CBIndex::c_3, tt::CBIndex::c_2, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();

    pack_tile(0, tt::CBIndex::c_16);
    tile_regs_release();

    cb_pop_front(tt::CBIndex::c_2, 1);
    cb_pop_front(tt::CBIndex::c_3, 1);
    cb_push_back(tt::CBIndex::c_16, 1);
    }
}
}
