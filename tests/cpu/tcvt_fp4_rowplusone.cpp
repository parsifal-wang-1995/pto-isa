/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr uint8_t CANARY = 0xa5;
int checks = 0;
void require(bool condition, const char* message)
{
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

// Decode the FP4 value and encode normalized E4M3 numerically, independently
// of the production byte LUT and the simulator's MXType conversions.
uint8_t reference(uint8_t nibble)
{
    const int exponent = (nibble >> 1) & 3;
    const int mantissa = nibble & 1;
    const double magnitude = exponent == 0 ? 0.5 * mantissa : std::ldexp(2.0 + mantissa, exponent - 2);
    const uint8_t sign = (nibble & 8) ? 0x80 : 0;
    if (magnitude == 0)
        return sign;
    int e;
    const double fraction = std::frexp(magnitude, &e);
    return sign | static_cast<uint8_t>(((e + 6) << 3) | std::lround((fraction * 2.0 - 1.0) * 8.0));
}

template <int SR, int SC, int DR, int DC, CompactMode Compact, bool NZ = true>
void checkCase(unsigned rows, unsigned cols, RoundMode round = RoundMode::CAST_RINT, int api = 0)
{
    using Source = Tile<TileType::Vec, float4_e2m1x2_t, SR, SC, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using Dest = Tile<
        TileType::Vec, float8_e4m3_t, DR, DC, NZ ? BLayout::ColMajor : BLayout::RowMajor, DYNAMIC, DYNAMIC,
        NZ ? SLayout::RowMajor : SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Null, Compact>;
    Source source(rows, cols);
    Dest dest(rows, cols);
#ifndef __PTO_AUTO__
    TASSIGN(source, 0);
    TASSIGN(dest, 65536 + 32);
    auto* guards = reinterpret_cast<uint8_t*>(dest.data()) - 32;
    std::fill_n(guards, Dest::StorageBytes + 64, CANARY);
#endif
    auto* input = reinterpret_cast<uint8_t*>(source.data());
    auto* output = reinterpret_cast<uint8_t*>(dest.data());
    std::fill_n(output, Dest::StorageBytes, CANARY);
    for (size_t i = 0; i < Source::StorageBytes; ++i)
        input[i] = static_cast<uint8_t>(i * 37 + i / 256);
    const std::vector<uint8_t> before(input, input + Source::StorageBytes);
    const size_t pitch =
        Compact == CompactMode::Null ? DR : ((rows + 15) / 16) * 16 + (Compact == CompactMode::RowPlusOne ? 1 : 0);
    std::vector<uint8_t> expected(Dest::StorageBytes, CANARY);
    std::array<bool, 256> seen{};
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned col = 0; col < cols; ++col) {
            const uint8_t packed = input[row * (SC / 2) + col / 2];
            seen[packed] = true;
            const uint8_t code = (col & 1) ? packed >> 4 : packed & 15;
            const size_t offset = NZ ? (col / 32) * pitch * 32 + row * 32 + col % 32 : row * DC + col;
            require(offset < expected.size(), "reference exceeds storage capacity");
            expected[offset] = reference(code);
        }
    }
    Tile<TileType::Vec, int32_t, 1, 8> unused;
    switch (api) {
        case 0:
            TCVT(dest, source, round);
            break;
        case 1:
            TCVT(dest, source, round, SaturationMode::ON);
            break;
        case 2:
            TCVT(dest, source, round, SaturationMode::OFF);
            break;
        case 3:
            TCVT(dest, source, unused, round);
            break;
        case 4:
            TCVT(dest, source, unused, round, SaturationMode::OFF);
            break;
        default:
            TCVT<false>(dest, source, round, SaturationMode::ON);
            break;
    }
    const auto mismatch = std::mismatch(expected.begin(), expected.end(), output);
    if (mismatch.first != expected.end()) {
        const size_t byte = static_cast<size_t>(mismatch.first - expected.begin());
        std::cerr << "rows=" << rows << " cols=" << cols << " pitch=" << pitch << " byte=" << byte
                  << " expected=" << unsigned(expected[byte]) << " actual=" << unsigned(output[byte]) << '\n';
    }
    require(std::equal(expected.begin(), expected.end(), output), "raw output, invalid region or NZ gap differs");
    require(std::equal(before.begin(), before.end(), input), "TCVT modified source");
    if (rows * cols >= 4096 && cols == SC) {
        require(
            std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }), "not all 256 packed byte codes covered");
    }
    if (rows && cols) {
        require(dest.GetElement(0, 0).RawData() == expected[0], "CPU object-aware NZ indexing differs");
    }
#ifndef __PTO_AUTO__
    require(std::all_of(guards, guards + 32, [](uint8_t x) { return x == CANARY; }), "leading guard changed");
    require(
        std::all_of(
            output + Dest::StorageBytes, output + Dest::StorageBytes + 32, [](uint8_t x) { return x == CANARY; }),
        "trailing guard changed");
#endif
}

template <TInsertMode Mode>
void checkTransport()
{
    using Source = Tile<TileType::Vec, float4_e2m1x2_t, 32, 128, BLayout::RowMajor, 19, 93>;
    using Padded = Tile<
        TileType::Vec, float8_e4m3_t, 32, 128, BLayout::ColMajor, 19, 93, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using Matrix = Tile<TileType::Mat, float8_e4m3_t, 64, 160, BLayout::ColMajor, 64, 160, SLayout::RowMajor>;
    Source source;
    Padded padded;
    Matrix matrix;
#ifndef __PTO_AUTO__
    TASSIGN(source, 0);
    TASSIGN(padded, 32768);
    TASSIGN(matrix, 0);
#endif
    auto* input = reinterpret_cast<uint8_t*>(source.data());
    auto* pad = reinterpret_cast<uint8_t*>(padded.data());
    auto* output = reinterpret_cast<uint8_t*>(matrix.data());
    for (size_t i = 0; i < Source::StorageBytes; ++i)
        input[i] = static_cast<uint8_t>(i * 37);
    std::fill_n(pad, Padded::StorageBytes, CANARY);
    std::fill_n(output, Matrix::StorageBytes, 0x7f);
    TCVT(padded, source, RoundMode::CAST_RINT);
    TINSERT<Mode>(matrix, padded, 16, 32);
    // Existing DMA moves aligned row/column tails as well as valid values.
    for (unsigned row = 0; row < 64; ++row) {
        for (unsigned col = 0; col < 160; ++col) {
            uint8_t expected = 0x7f;
            if (row >= 16 && row < 48 && col >= 32 && col < 128) {
                expected = CANARY;
                if (row < 35 && col < 125) {
                    const unsigned r = row - 16, c = col - 32;
                    const uint8_t packed = input[r * 64 + c / 2];
                    expected = reference((c & 1) ? packed >> 4 : packed & 15);
                }
            }
            const size_t offset = (col / 32) * 64 * 32 + row * 32 + col % 32;
            require(output[offset] == expected, "RowPlusOne TINSERT failed to remove exactly one padding row");
        }
    }
}

void checkStorage()
{
    using Full = Tile<
        TileType::Vec, float8_e4m3_t, 128, 256, BLayout::ColMajor, 128, 256, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    using Tail = Tile<
        TileType::Vec, float8_e4m3_t, 17, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
    static_assert(Full::Rows == 128 && Full::Cols == 256 && Full::Numel == 32768);
    static_assert(Full::StorageNumel == 33024 && Full::StorageBytes == 33024);
    static_assert(Tail::StorageBytes == 33 * 256);
    Full full;
#ifdef __PTO_AUTO__
    // Exercise both const and mutable automatic allocation before the last byte.
    const Full& cfull = full;
    require(cfull.data() != nullptr, "const automatic allocation failed");
#else
    TASSIGN(full, 0);
#endif
    full.data()[Full::StorageNumel - 1] = float8_e4m3_t::FromRaw(0x80);
    require(full.data()[Full::StorageNumel - 1].RawData() == 0x80, "automatic padded capacity missing");
}
} // namespace

int main(int argc, char** argv)
{
    NPU_MEMORY_INIT(NPUArch::A5);
#if defined(TCVT_FP4_REJECT)
    using S = Tile<TileType::Vec, float4_e2m1x2_t, 32, 128>;
#if TCVT_FP4_REJECT == 1
    using D = Tile<
        TileType::Vec, float8_e4m3_t, 17, 128, BLayout::ColMajor, 17, 128, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
#elif TCVT_FP4_REJECT == 2
    using D = Tile<
        TileType::Vec, float8_e4m3_t, 32, 128, BLayout::RowMajor, 32, 128, SLayout::NoneBox, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
#elif TCVT_FP4_REJECT == 3
    using D = Tile<
        TileType::Vec, float8_e4m3_t, 65536, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
#elif TCVT_FP4_REJECT == 4
    using D = Tile<
        TileType::Vec, float8_e4m3_t, 32, 128, BLayout::ColMajor, 31, 128, SLayout::RowMajor, 512, PadValue::Null,
        CompactMode::RowPlusOne>;
#endif
    S source;
    D destination;
    TCVT(destination, source, RoundMode::CAST_RINT);
#else
    if (argc > 1 && std::string(argv[1]) == "--invalid-runtime") {
        using S = Tile<TileType::Vec, float4_e2m1x2_t, 32, 128, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
        using D = Tile<
            TileType::Vec, float8_e4m3_t, 32, 128, BLayout::ColMajor, DYNAMIC, DYNAMIC, SLayout::RowMajor, 512,
            PadValue::Null, CompactMode::RowPlusOne>;
        S source(19, 93);
        D destination(18, 93);
        TCVT(destination, source, RoundMode::CAST_RINT);
        return 3;
    }
    checkStorage();
    require(reference(8) == 0x80, "negative zero reference lost sign");
    for (unsigned rows : {0u, 1u, 3u, 15u, 16u, 17u, 19u, 31u, 32u, 63u, 64u, 127u, 128u}) {
        for (unsigned cols : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u}) {
            checkCase<128, 256, 128, 256, CompactMode::RowPlusOne>(rows, cols);
        }
    }
    for (int round = 0; round <= 6; ++round) {
        for (int api = 0; api < 6; ++api) {
            checkCase<32, 512, 32, 512, CompactMode::RowPlusOne>(32, 512, static_cast<RoundMode>(round), api);
        }
    }
    checkCase<5, 640, 32, 704, CompactMode::RowPlusOne>(3, 579);
    checkCase<3, 1088, 32, 1152, CompactMode::RowPlusOne>(2, 1025);
    checkCase<1, 64, 16, 96, CompactMode::RowPlusOne>(1, 1);
    checkCase<20, 128, 32, 192, CompactMode::RowPlusOne>(19, 128);
    checkCase<2, 512, 32, 544, CompactMode::RowPlusOne>(1, 257);
    checkCase<5, 640, 32, 704, CompactMode::Normal>(3, 579);
    checkCase<5, 640, 32, 704, CompactMode::Null>(3, 579);
    checkCase<5, 640, 7, 704, CompactMode::Null, false>(3, 579);
    checkTransport<TInsertMode::SPLIT2>();
    checkTransport<TInsertMode::SPLIT4>();
    std::cout << "FP4 RowPlusOne checks passed: " << checks << '\n';
#endif
}
