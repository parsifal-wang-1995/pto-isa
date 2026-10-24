/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TCVT_FP4_HPP
#define PTO_TCVT_FP4_HPP

#include <pto/common/pto_tile.hpp>
#if defined(__CPU_SIM)
#include <cassert>
#define PTO_FP4_ASSERT(condition, message) assert((condition) && (message))
#else
#define PTO_FP4_ASSERT(condition, message) PTO_ASSERT(condition, message)
#endif

namespace pto {
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void checkFp4E2M1ToE4M3(TileDataD& dst, TileDataS& src, uint32_t rows, uint32_t cols)
{
    static_assert(
        TileDataS::Loc == TileType::Vec && TileDataD::Loc == TileType::Vec,
        "TCVT FP4 E2M1 -> FP8 E4M3 requires Vec source and destination.");
    static_assert(
        TileDataS::isRowMajor && TileDataS::SFractal == SLayout::NoneBox && TileDataS::Compact == CompactMode::Null,
        "TCVT FP4 E2M1 -> FP8 E4M3 source must be ND with CompactMode::Null.");
    static_assert(
        TileDataS::Cols >= 64 && TileDataS::Cols % 64 == 0,
        "TCVT FP4 E2M1 -> FP8 E4M3 source rows require a multiple of 64 logical columns.");
    constexpr bool dstND = TileDataD::isRowMajor && TileDataD::SFractal == SLayout::NoneBox;
    constexpr bool dstNZ = !TileDataD::isRowMajor && TileDataD::SFractal == SLayout::RowMajor &&
                           TileDataD::SFractalSize == TileConfig::fractalABSize;
    static_assert(dstND || dstNZ, "TCVT FP4 E2M1 -> FP8 E4M3 destination must be ND or standard 512-byte NZ.");
    static_assert(
        TileDataD::Compact == CompactMode::Null ||
            (dstNZ && (TileDataD::Compact == CompactMode::Normal || TileDataD::Compact == CompactMode::RowPlusOne)),
        "TCVT FP4 E2M1 -> FP8 E4M3 supports Null, or Normal/RowPlusOne for NZ.");
    static_assert(
        !dstNZ || TileDataD::Rows % FRACTAL_NZ_ROW == 0, "TCVT FP8 NZ physical rows must be a multiple of 16.");
    constexpr uint64_t maxPanelRows =
        static_cast<uint64_t>(TileDataD::Rows) + (TileDataD::Compact == CompactMode::RowPlusOne ? 1u : 0u);
    static_assert(!dstNZ || maxPanelRows <= 0xFFFF, "TCVT FP8 NZ row pitch must fit the 16-bit block stride.");
    static_assert(
        static_cast<uint64_t>(TileDataS::Rows) * (TileDataS::Cols / 2) <= 0x7FFFFFFF &&
            TileDataD::StorageBytes <= 0x7FFFFFFF,
        "TCVT FP4 E2M1 -> FP8 E4M3 byte offsets must fit signed 32 bits.");
    static_assert(
        TileDataD::ValidRow == DYNAMIC || TileDataS::ValidRow == DYNAMIC || TileDataD::ValidRow == TileDataS::ValidRow,
        "TCVT FP4 E2M1 -> FP8 E4M3 valid rows must agree.");
    static_assert(
        TileDataD::ValidCol == DYNAMIC || TileDataS::ValidCol == DYNAMIC || TileDataD::ValidCol == TileDataS::ValidCol,
        "TCVT FP4 E2M1 -> FP8 E4M3 logical valid columns must agree.");
    PTO_FP4_ASSERT(
        rows == dst.GetValidRow() && cols == dst.GetValidCol() && rows == src.GetValidRow() &&
            cols == src.GetValidCol(),
        "TCVT FP4/FP8 valid regions must agree.");
    PTO_FP4_ASSERT(
        rows <= TileDataS::Rows && rows <= TileDataD::Rows && cols <= TileDataS::Cols && cols <= TileDataD::Cols,
        "TCVT FP4/FP8 valid region exceeds physical tile bounds.");
    static_assert(
        TileDataS::Rows <= 0xFFFF && TileDataD::Rows <= 0xFFFF && TileDataS::Cols / (2 * 256) <= 0xFFFF,
        "TCVT FP4/FP8 VF loop bounds must fit 16-bit counters.");
}
} // namespace pto
#undef PTO_FP4_ASSERT
#endif
