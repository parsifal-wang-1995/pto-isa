/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;

using Source = Tile<TileType::Vec, float4_e2m1x2_t, 128, 256>;
using Padded = Tile<
    TileType::Vec, float8_e4m3_t, 128, 256, BLayout::ColMajor, 128, 256, SLayout::RowMajor, 512, PadValue::Null,
    CompactMode::RowPlusOne>;
using Matrix = Tile<TileType::Mat, float8_e4m3_t, 128, 256, BLayout::ColMajor, 128, 256, SLayout::RowMajor>;
using Pipe = TPipe<3, Direction::DIR_V2C, 65536, 2>;
static_assert(Padded::Rows == 128 && Padded::Cols == 256 && Padded::Numel == 32768);
static_assert(Padded::StorageNumel == 33024 && Padded::StorageBytes == 33024);

__tf__ PTO_INTERNAL void exportLastStorageByte(Padded::TileDType __in__ padded, __gm__ uint8_t* output)
{
    output[0] = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(padded))[Padded::StorageBytes - 1];
}

// Compile this source in both explicit and __PTO_AUTO__ modes. The vector
// conversion and DMA/FIFO consumers must accept exactly the same padded type.
extern "C" __global__ AICORE void tcvtFp4RowPlusOneCompile(
    __gm__ float4_e2m1x2_t* input, __gm__ uint8_t* output, __gm__ void* fifo, bool push)
{
    Source source;
    Padded padded;
    Matrix matrix;
#ifndef __PTO_AUTO__
    TASSIGN(source, 0);
    TASSIGN(padded, Source::StorageBytes);
    TASSIGN(matrix, 0);
#endif
    GlobalTensor<float4_e2m1x2_t, Shape<1, 1, 1, 128, 256>, pto::Stride<32768, 32768, 32768, 256, 1>> global(input);
    TLOAD(source, global);
    TCVT(padded, source, RoundMode::CAST_RINT);
    TINSERT(matrix, padded, 0, 0);
    if (push) {
        Pipe pipe(fifo, 0, 0);
        TPUSH<Pipe, Padded, TileSplitAxis::TILE_NO_SPLIT>(pipe, padded);
    }
    // Force retention of the final physical byte in the tile_size allocation.
    exportLastStorageByte(padded.data(), output);
}
