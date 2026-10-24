#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile and run the public CPU TCVT RowPlusOne contract, one compiler at a time."""

import argparse
import json
from pathlib import Path
import resource
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / "tests/cpu/tcvt_fp4_rowplusone.cpp"
    base = [args.compiler, "-std=c++23", "-D__CPU_SIM", "-I" + str(root / "include")]
    records = []

    def command(name, argv, expected=0, needle=None):
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        text = result.stdout + result.stderr
        (output / (name + ".log")).write_text(text)
        passed = result.returncode == expected if expected is not None else result.returncode != 0
        passed = passed and (needle is None or needle in text)
        records.append({"name": name, "argv": list(map(str, argv)), "exit": result.returncode, "passed": passed})
        (output / "result.json").write_text(json.dumps(records, indent=2) + "\n")
        if not passed:
            raise RuntimeError(f"{name} failed: exit={result.returncode}; see {output / (name + '.log')}")

    for mode in ("manual", "auto"):
        executable = output / ("tcvt-fp4-" + mode)
        flags = ["-D__PTO_AUTO__"] if mode == "auto" else []
        command(mode + "-compile", base + flags + [str(source), "-o", str(executable)])
        command(mode + "-run", [str(executable)])
        command(mode + "-reject-runtime", [str(executable), "--invalid-runtime"], None, "valid regions must agree")
    messages = {
        1: "physical rows must be a multiple of 16",
        2: "supports Null, or Normal/RowPlusOne for NZ",
        3: "row pitch must fit the 16-bit block stride",
        4: "valid rows must agree",
    }
    for case, message in messages.items():
        command(f"reject-{case}", base + [f"-DTCVT_FP4_REJECT={case}", "-fsyntax-only", str(source)], None, message)
    print(json.dumps({"passed": len(records), "output": str(output), "checks": records}, indent=2))


if __name__ == "__main__":
    main()
