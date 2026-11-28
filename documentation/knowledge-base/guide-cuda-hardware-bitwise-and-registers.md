---
title: "CUDA hardware foundations: registers, clock cycles, bitwise arithmetic, and data types"
description: >
  Comprehensive pedagogical guide to CUDA hardware foundations, 32-bit register anatomy,
  instruction latencies across arithmetic vs memory units, bitwise operations from first
  principles, hexadecimal notation, sign-extension pitfalls, float bit-reinterpretation,
  core intrinsics (__popc, __clz, __ffs, __brev), and architectural analysis of discarded
  compaction alternatives.
created: 2026-11-09
date_created: 2026-11-09
date_changed: 2026-11-09
status: final
author:
  - "Lucas Galdino"
  - "Antigravity"
type: guide
scope: global
modifications:
  - date_modified: 2026-11-09
    modifications:
      - description: >
          Created modular hardware foundation guide extracted and deepened from monolithic
          warp-primitives reference, answering undergraduate questions on bitwise arithmetic,
          clock cycle latencies, register starvation, and iso-algorithmic ordering contracts.
related_files:
  - "[guide-cuda-warp-primitives-and-intrinsics.md](./guide-cuda-warp-primitives-and-intrinsics.md)"
  - "[analysis-crossover-matrix-formulation-and-mechanics.md](./analysis-crossover-matrix-formulation-and-mechanics.md)"
  - "[analysis-crossover-gpu-architecture-and-scaling.md](./analysis-crossover-gpu-architecture-and-scaling.md)"
  - "[analysis-gpu-ga-strategies-coalescing-and-divergence.md](./analysis-gpu-ga-strategies-coalescing-and-divergence.md)"
  - "[analysis-gpu-memory-latency-and-occupancy.md](./analysis-gpu-memory-latency-and-occupancy.md)"
tags:
  - guide/cuda
  - guide/architecture
  - notes/cuda
  - notes/hardware
  - analysis/architecture
  - analysis/performance
  - algorithm/crossover
  - algorithm/genetic-algorithm
  - algorithm/tsp
  - benchmark/gpu
---

# CUDA hardware foundations: registers, clock cycles, bitwise arithmetic, and data types

- [CUDA hardware foundations: registers, clock cycles, bitwise arithmetic, and data types](#cuda-hardware-foundations-registers-clock-cycles-bitwise-arithmetic-and-data-types)
  + [1. Executive overview](#1-executive-overview)
  + [2. Bitwise operations from zero: the binary foundation](#2-bitwise-operations-from-zero-the-binary-foundation)
    - [2.1 Binary representation, powers of two, and hexadecimal notation](#21-binary-representation-powers-of-two-and-hexadecimal-notation)
      + [Powers of Two](#powers-of-two)
      + [Why Hexadecimal (`0x...`)?](#why-hexadecimal-0x)
      + [Demystifying `0xFFFFFFFFu`:](#demystifying-0xffffffffu)
    - [2.2 Fundamental bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)](#22-fundamental-bitwise-operators------)
    - [2.3 The unified 4-trick example: tracking City 75 in hardware](#23-the-unified-4-trick-example-tracking-city-75-in-hardware)
      + [Step 1: Find the Word Index (`c >> 5`)](#step-1-find-the-word-index-c--5)
      + [Step 2: Find the Bit Position inside that Word (`c & 31`)](#step-2-find-the-bit-position-inside-that-word-c--31)
      + [Step 3: Create a Probe Mask to Set or Test Bit 11 (`1u << k`)](#step-3-create-a-probe-mask-to-set-or-test-bit-11-1u--k)
      + [Step 4: Create "Blinders" to Look Only at Lower Cities (`(1u << k) - 1u`)](#step-4-create-blinders-to-look-only-at-lower-cities-1u--k---1u)
  + [3. GPU clock cycles, instruction latency, and execution pipeline starvation](#3-gpu-clock-cycles-instruction-latency-and-execution-pipeline-starvation)
    - [3.1 What is a clock cycle in hardware?](#31-what-is-a-clock-cycle-in-hardware)
    - [3.2 Why latency varies by operation type (and which takes the most time)](#32-why-latency-varies-by-operation-type-and-which-takes-the-most-time)
    - [3.3 Pipeline stalls: does the register stay idle when no warps are ready?](#33-pipeline-stalls-does-the-register-stay-idle-when-no-warps-are-ready)
  + [4. GPU physical registers, data types width matrix, and type safety](#4-gpu-physical-registers-data-types-width-matrix-and-type-safety)
    - [4.1 Physical anatomy of a 32-bit register file (wires and bits)](#41-physical-anatomy-of-a-32-bit-register-file-wires-and-bits)
    - [4.2 Data types width matrix: 8-bit to 64-bit integers and floats](#42-data-types-width-matrix-8-bit-to-64-bit-integers-and-floats)
    - [4.3 The signed 8-bit sign-extension trap: why `__popc(-5)` returns 31 instead of 7](#43-the-signed-8-bit-sign-extension-trap-why-__popc-5-returns-31-instead-of-7)
    - [4.4 SIMD Within A Register (SWAR) for 4-byte batch loads](#44-simd-within-a-register-swar-for-4-byte-batch-loads)
    - [4.5 Floating-point bitwise reinterpretation (`__float_as_uint`): using floats masked as ints](#45-floating-point-bitwise-reinterpretation-__float_as_uint-using-floats-masked-as-ints)
      + [Why is this immensely useful in GPU optimization?](#why-is-this-immensely-useful-in-gpu-optimization)
  + [5. Core integer bit-manipulation intrinsics (`__popc`, `__clz`, `__ffs`, `__brev`)](#5-core-integer-bit-manipulation-intrinsics-__popc-__clz-__ffs-__brev)
    - [5.1 Summary table and mathematical definitions](#51-summary-table-and-mathematical-definitions)
    - [5.2 Detailed visual traces and TSP solver applications](#52-detailed-visual-traces-and-tsp-solver-applications)
      + [1. `__popc`: Stream Compaction and Active Thread Count](#1-__popc-stream-compaction-and-active-thread-count)
      + [2. `__clz`: Dynamic Reduction Step Sizes and $\\lfloor \\log\_2(x) \\rfloor$](#2-__clz-dynamic-reduction-step-sizes-and-lfloor-log_2x-rfloor)
      + [3. `__ffs`: Finding the First Improving 2-Opt Move](#3-__ffs-finding-the-first-improving-2-opt-move)
      + [4. `__brev`: Reversing Bitpacked Routes in Inversion Mutation](#4-__brev-reversing-bitpacked-routes-in-inversion-mutation)
  + [6. Hardware comparison of stream compaction alternatives](#6-hardware-comparison-of-stream-compaction-alternatives)
    - [6.1 Why `atomicAdd` in Global or Shared Memory scrambles tour ordering](#61-why-atomicadd-in-global-or-shared-memory-scrambles-tour-ordering)
    - [6.2 Is there a problem with $P\_2\[5\]$ being placed before $P\_2\[2\]$?](#62-is-there-a-problem-with-p_25-being-placed-before-p_22)
    - [6.3 Latency comparison across all 5 compaction mechanisms](#63-latency-comparison-across-all-5-compaction-mechanisms)

---

## 1. Executive overview

Writing high-performance CUDA kernels for discrete combinatorial optimization (like the Traveling Salesperson Problem and Genetic Algorithms) requires mastering the physical reality of the GPU: registers, clock cycles, bitwise operations, and memory transactions.

Unlike high-level python or standard C++ where variables exist as abstract objects, every variable in a GPU Streaming Multiprocessor (SM) physically sits on **32 metal wires** inside a register file. Understanding how these wires represent numbers, how bits are shifted in a single clock tick, and why memory access must follow hardware rules is essential to writing code that is both blisteringly fast and mathematically sound.

This guide provides the foundational hardware and bitwise knowledge required before exploring warp-level collectives. For warp vote and shuffle primitives, see the companion guide [guide-cuda-warp-primitives-and-intrinsics.md](./guide-cuda-warp-primitives-and-intrinsics.md).

---

## 2. Bitwise operations from zero: the binary foundation

### 2.1 Binary representation, powers of two, and hexadecimal notation

Computers store all information using electrical voltages: a high voltage represents `1`, and a low voltage represents `0`. A single digit is a **bit**.

#### Powers of Two

In standard decimal (base 10), each place value represents a power of 10: $10^0 = 1, 10^1 = 10, 10^2 = 100$.
In binary (base 2), each place value represents a power of 2:

$$\begin{array}{rcccccccc}
\text{Bit Position } k: & b_7 & b_6 & b_5 & b_4 & b_3 & b_2 & b_1 & b_0 \\
\text{Place Value } 2^k: & 2^7=128 & 2^6=64 & 2^5=32 & 2^4=16 & 2^3=8 & 2^2=4 & 2^1=2 & 2^0=1
\end{array}$$

For example, the number $75$ in binary is:

$$75 = 64 + 8 + 2 + 1 = 1 \cdot 2^6 + 0 \cdot 2^5 + 0 \cdot 2^4 + 1 \cdot 2^3 + 0 \cdot 2^2 + 1 \cdot 2^1 + 1 \cdot 2^0 = \texttt{0b01001011}$$

#### Why Hexadecimal (`0x...`)?
A 32-bit register holds 32 ones and zeros. Writing `0b11111111111111111111111111111111` is exhausting, unreadable, and error-prone.

Hexadecimal (base 16) solves this because $16 = 2^4$. **Exactly 4 bits in binary correspond to 1 character in hexadecimal**:

| Binary (4 bits) | Decimal | Hexadecimal |
| :---: | :---: | :---: |
| `0000` | 0 | `0x0` |
| `0001` | 1 | `0x1` |
| `0010` | 2 | `0x2` |
| `0011` | 3 | `0x3` |
| `0100` | 4 | `0x4` |
| `0101` | 5 | `0x5` |
| `0110` | 6 | `0x6` |
| `0111` | 7 | `0x7` |
| `1000` | 8 | `0x8` |
| `1001` | 9 | `0x9` |
| `1010` | 10 | `0xA` |
| `1011` | 11 | `0xB` |
| `1100` | 12 | `0xC` |
| `1101` | 13 | `0xD` |
| `1110` | 14 | `0xE` |
| `1111` | 15 | `0xF` |

#### Demystifying `0xFFFFFFFFu`:
When you see `0xFFFFFFFFu` in CUDA code:
- Each `F` represents four ones: `1111`.
- Eight `F`s represent $8 \times 4 = 32$ ones:
  $$\texttt{0xFFFFFFFF} = \texttt{0b}\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}\,\underbrace{\texttt{1111}}_{F}$$
- The trailing **`u`** stands for **unsigned**. In C++, integer literals default to signed 32-bit integers. In two's-complement arithmetic, 32 ones interpreted as a signed integer is $-1$. Appending `u` explicitly tells the compiler: *"Treat this as an unsigned integer with value $2^{32} - 1 = 4{,}294{,}967{,}295$, not $-1$."*

### 2.2 Fundamental bitwise operators (`&`, `|`, `^`, `~`, `<<`, `>>`)

Six fundamental bitwise operators run directly on the physical logic gates of the GPU ALU in **1 single clock cycle**:

| Operator | Name | Rule | Example ($A = \texttt{0b1100}, B = \texttt{0b1010}$) |
| :---: | :--- | :--- | :--- |
| `&` | **Bitwise AND** | Result is `1` only if **both** bits are `1` | `A & B = 0b1000` |
| `\|` | **Bitwise OR** | Result is `1` if **at least one** bit is `1` | `A \| B = 0b1110` |
| `^` | **Bitwise XOR** | Result is `1` if bits are **different** ($0 \ne 1$) | `A ^ B = 0b0110` |
| `~` | **Bitwise NOT** | Inverts all bits ($0 \to 1, 1 \to 0$) | `~A = ...0011` |
| `<<` | **Left Shift** | Shifts bits left by $k$ places (multiplies by $2^k$) | `A << 1 = 0b11000` ($12 \times 2 = 24$) |
| `>>` | **Right Shift** | Shifts bits right by $k$ places (divides by $2^k$) | `A >> 1 = 0b0110` ($\lfloor 12 / 2 \rfloor = 6$) |

### 2.3 The unified 4-trick example: tracking City 75 in hardware

In high-performance TSP solvers, we often encounter four bitwise expressions:
1. `c >> 5`
2. `c & 31`
3. `1u << k`
4. `(1u << k) - 1u`

To show why these exist and how they work together, consider a concrete problem:
> **Scenario**: We have a 100-city TSP tour ($n = 100$). We want to check whether **City 75** is present in a cut segment, using a compact 32-bit bitpacked array `uint32_t bitmap[4]`.

Here is the complete numerical walkthrough connecting all 4 tricks:

```
                  THE 100-CITY BITMAP ARRAY: uint32_t bitmap[4]
 Word Index:        [ Word 0 ]         [ Word 1 ]         [ Word 2 ]         [ Word 3 ]
 Cities Covered:   Cities 0..31       Cities 32..63      Cities 64..95      Cities 96..99
```

#### Step 1: Find the Word Index (`c >> 5`)
- Why `5`? Because $2^5 = 32$. There are 32 bits in each word.
- To find which word contains City 75, we need $\lfloor 75 / 32 \rfloor$.
- Normal integer division `75 / 32` requires a division circuit ($\sim 20$ clock cycles).
- Bit shift `75 >> 5` moves the bits 5 places to the right in **1 single clock cycle**:
  $$75 = \texttt{0b01001011} \implies 75 \gg 5 = \texttt{0b00000010} = \mathbf{2}$$
- **Result**: City 75 lives inside `bitmap[2]`!

#### Step 2: Find the Bit Position inside that Word (`c & 31`)
- Why `31`? Because $31 = 2^5 - 1 = \texttt{0b00011111}$ (five ones).
- In decimal, $147 \bmod 100 = 47$ (the last 2 digits).
- In binary, $c \bmod 32$ is simply the value of the **lowest 5 bits**!
- Masking with `31` (`75 & 31`) zeroes out the upper bits in **1 clock cycle**:
  $$\begin{array}{rll}
  75: & \texttt{0b01001011} \\
  \& \quad 31: & \texttt{0b00011111} \\
  \hline
  \text{Result}: & \texttt{0b00001011} = \mathbf{11} \quad (75 \bmod 32 = 11)
  \end{array}$$
- **Result**: City 75 is stored at **Bit 11** inside `bitmap[2]`!

#### Step 3: Create a Probe Mask to Set or Test Bit 11 (`1u << k`)
- We need a 32-bit number where only Bit 11 is `1`, and all other 31 bits are `0`.
- Taking `1u` (`0b0000...0001`) and shifting left by $k = 11$:
  $$\texttt{mask} = 1\text{u} \ll 11 = \texttt{0b00000000\_00000000\_00001000\_00000000} \quad (\text{Value } 2^{11} = 2048)$$
- To **mark** City 75 as present:
  ```cpp
  atomicOr(&bitmap[2], 1u << 11);
  ```
  The bitwise OR turns on Bit 11 without disturbing any other city in Word 2!
- To **test** if City 75 is already present:
  ```cpp
  bool is_present = (bitmap[2] >> 11) & 1;
  ```
  Shifting right by 11 brings Bit 11 down to Bit 0. The `& 1` extracts that single bit (`1` if present, `0` if absent).

#### Step 4: Create "Blinders" to Look Only at Lower Cities (`(1u << k) - 1u`)
- Suppose thread Lane $k = 11$ wants to count how many cities *below* index 11 are active.
- Subtracting $1$ from $2^{11}$ turns on all bits from 0 to 10:
  $$(1\text{u} \ll 11) - 1\text{u} = 2048 - 1 = 2047 = \texttt{0b00000000\_00000000\_00000111\_11111111}$$
- Notice: Exactly eleven ones! Bits $0 \dots 10$ are `1`, while Bit 11 and all higher bits are `0`.
- Masking with this value blocks out everything to the right, creating "blinders" for parallel prefix scans!

All 4 tricks harmonize together as a complete, unified toolkit for bitpacked manipulation.

---

## 3. GPU clock cycles, instruction latency, and execution pipeline starvation

### 3.1 What is a clock cycle in hardware?

Every Streaming Multiprocessor (SM) on the NVIDIA GeForce GTX 1050 (Pascal GP107) runs at base clock frequency $f_{\text{clock}} \approx 1.493\text{ GHz}$ ($1.493 \times 10^9\text{ cycles/second}$).

The physical duration of 1 clock tick is:

$$T_{\text{cycle}} = \frac{1}{1.493 \times 10^9\text{ s}^{-1}} \approx 0.67\text{ nanoseconds}$$

Inside the silicon chip, a clock cycle is an electrical pulse emitted by an oscillating crystal. On each pulse, electrons travel across microscopic transistor gates to update flip-flop circuits. Nothing in the SM can happen faster than 1 cycle.

### 3.2 Why latency varies by operation type (and which takes the most time)

When an instruction is issued, the number of clock ticks until its result is available in a destination register is called **Instruction Latency ($L$)**.

Why do different operations take different amounts of time?

```
                     INSTRUCTION LATENCY SPECTRUM ON PASCAL GP107
  ┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
  │   Simple 32-bit ALU  │   FP32 Multiply/FMA  │  Integer Div/Modulo  │  Global Memory DRAM  │
  │     4 to 6 cycles    │     4 to 6 cycles    │    20 to 35 cycles   │   200 to 600 cycles  │
  └──────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
  Fast on-chip logic ───────────────────────────────────────────────► Slow off-chip physical bus
```

1. **Simple 32-bit ALU (`add`, `sub`, bit shifts `<<`, bitwise `&`, `|`, `^`) — $4\text{ to }6\text{ cycles}$**:
   These circuits consist of only a few layers of NAND/NOR logic gates physically etched directly next to the register file. The electrical signals propagate and settle in $4$ clock ticks.
2. **Floating-Point Multiplication & Fused Multiply-Add (`fmul`, `fma`) — $4\text{ to }6\text{ cycles}$**:
   Pipelined hardware multipliers compute $A \times B + C$ in high-speed hardware blocks.
3. **Integer Division and Modulo (`a / b`, `a % b`) — $20\text{ to }35\text{ cycles}$**:
   There is **no single-cycle hardware circuit** for general integer division! The GPU ALU must run an iterative subtract-and-shift microcode routine (non-restoring division). This is why replacing `c / 32` with `c >> 5` saves dozens of clock cycles per thread!
4. **Special Function Units (`__sinf`, `__cosf`, `__expf`, `__logf`, `rsqrt`) — $16\text{ to }20\text{ cycles}$**:
   Computed using hardware quadratic polynomial approximations in dedicated SFU hardware.
5. **Global Memory DRAM Access (`population[idx]`, `tour[city]`) — $200\text{ to }600\text{ cycles}$**:
   **This takes the most time by far!** When a thread requests data not present in L1/L2 cache, the request must travel over physical copper traces off the GPU die to external GDDR5 memory chips. Round-trip transit takes hundreds of nanoseconds ($200\text{–}600$ clock ticks).

### 3.3 Pipeline stalls: does the register stay idle when no warps are ready?

The user asked:
> *"If an instruction takes $L$ cycles, does the register also stay idle if there are no other active warps?"*

**The answer is YES.**

NVIDIA GPUs tolerate long latencies through **Zero-Overhead Warp Scheduling**:
- On every clock cycle, the warp scheduler checks which warps are ready to execute an instruction.
- If Warp 0 is waiting $300$ cycles for a GDDR5 memory fetch, the scheduler instantly switches to Warp 1, then Warp 2, etc. Switching warps costs **0 clock cycles** because each warp has its own dedicated physical registers.
- **However, if there are no other active warps** (for example, if a kernel launched only 1 warp, or only a handful of threads):
  1. The Streaming Multiprocessor enters a state called **execution pipeline starvation**.
  2. The ALU execution units sit completely idle.
  3. The physical registers holding the waiting variables remain powered (holding their electrostatic charges), but **no computation occurs**.
  4. The GPU's massive computing power is thrown away while waiting for memory.

This is the exact reason why our old crossover kernel (which launched only $N=256$ threads across the whole GPU, occupying only 2 out of 5 SMs) suffered from terrible performance: with only 1 warp per block, whenever memory was accessed, the SM had no other warps to hide the latency, and the hardware sat stalled!

---

## 4. GPU physical registers, data types width matrix, and type safety

### 4.1 Physical anatomy of a 32-bit register file (wires and bits)

Inside each SM of the Pascal GP107 GPU, there is a physical Register File containing **65,536 registers**, each **exactly 32 bits wide**:

```
                       PHYSICAL 32-BIT REGISTER ANATOMY
  Wire:  b31 b30 b29 b28 b27 b26 b25 b24 ... b7  b6  b5  b4  b3  b2  b1  b0
        ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  R0:   │ 0 │ 1 │ 0 │ 0 │ 0 │ 0 │ 0 │ 0 │...│ 0 │ 0 │ 0 │ 0 │ 1 │ 0 │ 1 │ 1 │
        └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
         ▲ MSB (Bit 31: Sign bit for signed integers)                       ▲ LSB (Bit 0: Weight 2^0)
```

- Each bit physically corresponds to a flip-flop circuit capable of storing one binary state.
- **There are no 8-bit or 16-bit physical registers in NVIDIA GPUs!**
- If you declare a `uint8_t x` in CUDA, the GPU still allocates a full **32-bit register** (`R0`) for it. The upper 24 bits are simply filled with zeros.

### 4.2 Data types width matrix: 8-bit to 64-bit integers and floats

| C++ / CUDA Type | Width | Storage in GPU Register File | How to Count Bits Correctly | Hardware Instruction Issued |
| :--- | :---: | :--- | :--- | :--- |
| `uint32_t` | 32-bit | Exactly 1 physical register | `__popc(x)` | 1 `POPC` ALU instruction |
| `int32_t` | 32-bit | Exactly 1 physical register | `__popc((unsigned int)x)` | 1 `POPC` ALU instruction |
| `uint64_t` | 64-bit | **2 consecutive registers** (`R0`, `R1`) | `__popcll(x)` | 2 `POPC` + 1 `IADD` ALU instructions |
| `int64_t` | 64-bit | **2 consecutive registers** (`R0`, `R1`) | `__popcll((unsigned long long)x)` | 2 `POPC` + 1 `IADD` ALU instructions |
| `uint16_t` | 16-bit | 1 register (upper 16 bits zeroed) | `__popc((unsigned int)x)` | 1 `POPC` ALU instruction |
| `int16_t` | 16-bit | 1 register (sign extended) | `__popc((unsigned int)(uint16_t)x)` | 1 mask + 1 `POPC` instruction |
| `uint8_t` | 8-bit | 1 register (upper 24 bits zeroed) | `__popc((unsigned int)x)` | 1 `POPC` ALU instruction |
| `int8_t` | 8-bit | 1 register (sign extended) | `__popc((unsigned int)(uint8_t)x)` | 1 mask + 1 `POPC` instruction |
| `float` | 32-bit | 1 register (IEEE-754 bit pattern) | `__popc(__float_as_uint(x))` | 0-cycle cast + 1 `POPC` instruction |
| `double` | 64-bit | **2 consecutive registers** (`R0`, `R1`) | `__popcll(__double_as_ull(x))` | 0-cycle cast + 2 `POPC` + 1 `IADD` |

### 4.3 The signed 8-bit sign-extension trap: why `__popc(-5)` returns 31 instead of 7

Why does calling `__popc` on a negative signed `int8_t` produce a catastrophic bug?

Suppose `int8_t val = -5`.
In 8-bit two's-complement binary:

$$-5 = -128 + 64 + 32 + 16 + 8 + 2 + 1 = \texttt{0b11111011} \quad (\text{exactly 7 set bits})$$

However, `__popc` takes a **32-bit unsigned integer**.
When C++ passes an 8-bit signed integer into a 32-bit function, the CPU and GPU perform **sign extension** to ensure the 32-bit integer still numerically equals $-5$.
It copies the sign bit (`1`) across all 24 upper wires:

```
  1. True 8-bit representation of -5:
     [ 1 1 1 1 1 0 1 1 ]  --> Exactly 7 set bits

  2. Implicit C++ Sign Extension into 32-bit register:
     [ 11111111 11111111 11111111 ] [ 1 1 1 1 1 0 1 1 ]
       ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲    ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
       24 EXTRA SIGN-EXTENSION ONES!   Original 7 ones
     __popc(-5) returns: 24 + 7 = 31 (CATASTROPHIC BUG!)

  3. Correct Zero-Extended Cast: __popc((uint8_t)val)
     [ 00000000 00000000 00000000 ] [ 1 1 1 1 1 0 1 1 ]
       ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲    ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
       24 zeros                        Original 7 ones
     __popc((uint8_t)val) returns: 7 (CORRECT!)
```

**Rule**: Never pass a signed 8-bit or 16-bit integer to `__popc` without casting to `uint8_t` or `uint16_t` first!

### 4.4 SIMD Within A Register (SWAR) for 4-byte batch loads

The user asked about this tip:
> *"How is `reinterpret_cast<const uint32_t*>(byte_array)[idx]` a tip? What does that inline code do?"*

Normally, if a thread needs to read 4 consecutive `uint8_t` flags (e.g. checking whether cities 0, 1, 2, 3 are in the segment):
```cpp
uint8_t b0 = in_segment[0]; // 1 memory read (1 byte)
uint8_t b1 = in_segment[1]; // 1 memory read (1 byte)
uint8_t b2 = in_segment[2]; // 1 memory read (1 byte)
uint8_t b3 = in_segment[3]; // 1 memory read (1 byte)
```
This issues **4 separate memory instructions**.

**The SWAR Tip**:
Since 4 bytes equals 32 bits, and each register is 32 bits, we can read all 4 bytes simultaneously in **1 single 32-bit memory transaction**:
```cpp
// SWAR: Load 4 bytes at once in 1 memory instruction!
uint32_t four_cities = reinterpret_cast<const uint32_t*>(in_segment)[0];
```
Inside the 32-bit register:
- Bits 0..7 hold City 0
- Bits 8..15 hold City 1
- Bits 16..23 hold City 2
- Bits 24..31 hold City 3

If you want to count how many of those 4 cities are marked, you do:
```cpp
int count = __popc(four_cities);
```

> [!NOTE]
> **Why `__popc` counts marked cities correctly here:**
> In our lookup table, each flag is strictly a boolean value $\in \{0, 1\}$:
> - An unmarked city byte is `0x00` (`0b00000000`, containing 0 set bits).
> - A marked city byte is `0x01` (`0b00000001`, containing exactly 1 set bit).
>
> Because each marked city contributes exactly **one `1` bit** to the 32-bit register, `__popc(four_cities)` computes the exact number of active cities across all 4 slots in **1 single clock cycle**!
> If the array contained arbitrary integers (like `255`), `__popc` would count all 8 bits of that byte; the trick works specifically because of the discrete $\{0, 1\}$ boolean invariant. That is why it is called SIMD Within A Register (SWAR).

### 4.5 Floating-point bitwise reinterpretation (`__float_as_uint`): using floats masked as ints

The user asked:
> *"I can use floats masked as ints? How does that work and why is it useful?"*

In C++, casting a float to an int (`(int)3.5f`) performs a **numerical conversion**: it discards the `.5` and outputs integer `3`. The bits in the register are completely rewritten.

However, a 32-bit float (`float`) physically occupies 32 bits according to the IEEE-754 standard:
- **Bit 31**: Sign bit (`0` = positive, `1` = negative).
- **Bits 23..30**: Exponent (8 bits).
- **Bits 0..22**: Mantissa / Fraction (23 bits).

CUDA provides a special hardware intrinsic: `__float_as_uint(x)`:
```cpp
float my_float = -3.5f;
uint32_t raw_bits = __float_as_uint(my_float);
```
`__float_as_uint` does **NOT convert the number**. It tells the compiler:
> *"Leave the 32 physical wires of register `R0` completely untouched. Just allow me to read those wires as an unsigned integer."*
This instruction takes **0 clock cycles overhead**!

#### Why is this immensely useful in GPU optimization?
1. **Testing if a float is negative in 1 clock cycle**:
   Normally: `if (delta < 0.0f)` requires a floating-point comparison instruction.
   Bitwise: `if (__float_as_uint(delta) >> 31)` simply checks wire 31!
2. **Absolute value in 1 clock cycle**:
   To get `abs(x)`, simply clear bit 31 using an integer AND mask:
   ```cpp
   float fast_abs = __uint_as_float(__float_as_uint(x) & 0x7FFFFFFFu);
   ```
3. **Warp Shuffle across threads**:
   On older CUDA architectures, `__shfl_sync` only supported 32-bit integers. To exchange a float between threads, developers cast `__float_as_uint`, shuffled the 32 bits through the register crossbar, and cast back with `__uint_as_float` without losing a single bit of floating-point precision!

---

## 5. Core integer bit-manipulation intrinsics (`__popc`, `__clz`, `__ffs`, `__brev`)

### 5.1 Summary table and mathematical definitions

NVIDIA hardware provides 4 dedicated bitwise instructions implemented directly in the SM ALU:

| Intrinsic Signature | Hardware Instruction | Mathematical Meaning | Example on `0b00101100` ($44_{10}$) |
| :--- | :---: | :--- | :--- |
| `int __popc(unsigned int x)` | `POPC` | Population Count: Sum of set bits $\sum_{k=0}^{31} b_k$ | `__popc(44)` $\implies \mathbf{3}$ (bits 2, 3, 5 are `1`) |
| `int __clz(int x)` | `FLO.CLZ` | Count Leading Zeros: Number of zeros from MSB down | `__clz(44)` $\implies \mathbf{26}$ ($31 - 5 = 26$) |
| `int __ffs(int x)` | `FFS` | Find First Set: Position of lowest `1` (1-indexed; 0 if $x=0$) | `__ffs(44)` $\implies \mathbf{3}$ (Bit 2 is lowest `1`: $2+1=3$) |
| `unsigned int __brev(unsigned int x)` | `BREV` | Bit Reversal: Mirrors all 32 bits ($b_k \leftrightarrow b_{31-k}$) | Reverses entire 32-bit register |

For 64-bit integers (`uint64_t`), use the suffix `ll`: `__popcll(x)`, `__clzll(x)`, `__ffsll(x)`, `__brevll(x)`.

### 5.2 Detailed visual traces and TSP solver applications

#### 1. `__popc`: Stream Compaction and Active Thread Count
- **What it does**: Counts how many bits are `1`.
- **TSP Application**: When 32 threads in a warp vote on candidate cities, `__ballot_sync` returns a 32-bit integer. Calling `__popc(ballot_mask)` tells you **instantly in 1 cycle** how many valid cities were found across all 32 threads, and determines each thread's exact write rank.

#### 2. `__clz`: Dynamic Reduction Step Sizes and $\lfloor \log_2(x) \rfloor$
- **What it does**: Counts how many consecutive zeros exist starting from the highest bit (Bit 31).
- **Mathematical formula**: For any positive integer $x$, $\lfloor \log_2(x) \rfloor = 31 - \texttt{\_\_clz}(x)$.
- **TSP Application**: When performing parallel tree reductions on non-power-of-two problem sizes, `__clz` determines the largest power of two $\le n$ in 1 cycle without expensive loops or lookup tables.

#### 3. `__ffs`: Finding the First Improving 2-Opt Move
- **What it does**: Scans from Bit 0 upward and returns the 1-based index of the first bit that is `1`. If $x = 0$, returns `0`.
- **TSP Application**: When 32 threads evaluate 2-opt swaps in parallel, multiple threads might find improving moves ($\Delta < 0$).
  `ballot = __ballot_sync(0xFFFFFFFFu, delta < 0);`
  Calling `winning_lane = __ffs(ballot) - 1;` identifies the **first winning thread** in 1 cycle! Lane 0 can then broadcast that winning move to the rest of the warp.

#### 4. `__brev`: Reversing Bitpacked Routes in Inversion Mutation
- **What it does**: Mirrors the entire register: Bit 0 swaps with Bit 31, Bit 1 swaps with Bit 30, etc.
- **TSP Application**: When routes or subpaths are bitpacked, reversing a tour segment can be performed by mirroring bit chunks via `__brev` in 1 single clock cycle, instead of iterating over individual bits!

---

## 6. Hardware comparison of stream compaction alternatives

### 6.1 Why `atomicAdd` in Global or Shared Memory scrambles tour ordering

In Order Crossover, surviving candidate cities from Parent 2 must be written into `child`.
Why can't each thread just use an atomic counter (`atomicAdd(&counter, 1)`) to claim its write slot?

Let us inspect the physical failure modes of the alternatives:

```cpp
// DISCARDED ATTEMPT: Shared memory atomic counter
if (is_valid) {
    int slot = atomicAdd(&s_counter, 1);
    child[slot] = city;
}
```

When 4 threads simultaneously execute `atomicAdd(&s_counter, 1)`:
1. The 4 requests arrive at the Shared Memory Arbiter.
2. The arbiter must serialize them. Which thread wins the first slot ($0$), and which gets the second ($1$)?
3. **Arrival order is non-deterministic at the hardware level!** Due to microscopic clock jitter, wire capacitance, and bank conflicts, Thread 5 might reach the arbiter before Thread 2.
4. If Thread 5 wins, it gets Slot 0 and writes city $P_2[5] = F$.
5. Thread 2 arrives second, gets Slot 1 and writes city $P_2[2] = C$.

### 6.2 Is there a problem with $P_2[5]$ being placed before $P_2[2]$?

The user asked:
> *"Out of curiosity, is there any problem with $P_2[5]$ being placed before $P_2[2]$?"*

**The answer is an emphatic YES, for two critical reasons:**

1. **Destruction of the Genetic Algorithm Heuristic (Davis, 1985)**:
   The entire premise of Order Crossover (OX) is to preserve the **relative order of cities** from Parent 2.
   In Parent 2, City $C$ precedes City $F$ ($C \to \dots \to F$).
   If the GPU randomly swaps them ($F \to C$), you have broken the crossover heuristic and introduced random permutation noise.
2. **Violation of the Strict Iso-Algorithmic Bit-Identity Contract (`DP11-11`)**:
   Our CPU reference implementation `CrossoverCPUPermOrder._ox_core` strictly preserves Parent 2's ordering.
   If the GPU kernel produces $[F, C]$ on one run and $[C, F]$ on another:
   - The GPU kernel is **non-deterministic**.
   - Every unit test comparing CPU and GPU output will fail.
   - Chapter 4 of the thesis would compare apples to oranges, making empirical speedup claims scientifically invalid.

### 6.3 Latency comparison across all 5 compaction mechanisms

| Compaction Mechanism | Hardware Execution Unit | Latency (Clock Cycles) | Deterministic Ordering? | Status in Solver |
| :--- | :--- | :---: | :---: | :--- |
| **Global Memory `atomicAdd`** | GDDR5 DRAM / L2 Atomic Unit | **200–600 cycles** | ❌ No (scrambled) | **Discarded** |
| **Shared Memory `atomicAdd`** | On-chip SRAM Bank Arbiter | **30–120 cycles** | ❌ No (scrambled) | **Discarded** |
| **Hillis-Steele Tree Scan** | Shared memory + 5 barriers | **~120 cycles** | ✅ Yes | **Discarded** (too slow) |
| **Lane 0 Sequential Loop** | 1 thread executes 32 iterations | **64–128 cycles** | ✅ Yes | **Discarded** (31 threads idle) |
| **Warp `__ballot_sync` + `__popc`** | **SM Register File Crossbar** | **4–8 cycles** | ✅ **100% Deterministic** | **ACCEPTED (DP11-11)** |

The warp-synchronous approach (`__ballot_sync` + `__popc`) is the **only method** that combines single-digit clock cycle latency with 100% mathematical determinism and strict iso-algorithmic bit-identity!
