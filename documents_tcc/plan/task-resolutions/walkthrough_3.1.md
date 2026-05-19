---
title: "Walkthrough: Task 3.1 — Padronização para Inteiros Discretos (TSPLIB)"
task_id: "3.1"
description: "Conversão completa da aritmética de distâncias de float32 para int32 em 10 arquivos, corrigindo conformidade TSPLIB e bug de iso-algoritmicidade."
created: "2026-05-14 18:00 -0300"
author: "[[Lucas Galdino]]"
status: "done"
tags: [cuda, python, cupy, tsplib, metaheuristics, 2-opt, genetic-algorithm, bugfix, refactor, kernel, pipeline]
links:
  - "[plan-roteiro.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/plan-roteiro.md)"
  - "[knowledge_tsplib_arithmetic.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/knowledge-base/knowledge_tsplib_arithmetic.md)"
---

# Walkthrough: Task 3.1 — Padronização para Inteiros Discretos (TSPLIB)

## Objective

Converter toda a aritmética de distâncias de `float32` para `int32`, alinhando estritamente com a especificação TSPLIB (`nint(d) = floor(d + 0.5)`), e corrigir o bug de hardcoding de iterações 2-opt que invalidava a iso-algoritmicidade do benchmark.

---

## Context: Code ↔ Documentation ↔ Data Correlation

| Artifact | Location | Relevant Observation |
|---|---|---|
| Distância de saída | [matrix.py L229](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py#L229) | Cast `astype(xp.float32)` descartava precisão inteira já calculada em `int64` internamente (L133) |
| Kernel principal | [ga_fujimoto.cu L19](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu#L19) | `float*` para distâncias e `float` para fitness — inconsistente com TSPLIB |
| Kernel early-stop | [ga_fujimoto_early_stop.cu L229](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto_early_stop.cu#L229) | `two_opt_iterations` declarado na assinatura mas `50` hardcoded no corpo |
| 2-opt kernels | [two_opt_single.cu L56](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/two_opt_single.cu#L56) | Epsilon `1e-5f` em comparação — necessário apenas por imprecisão float |
| Wrapper FullGPU | [genetic_algorithm_full_gpu_early_stop.py L115](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_full_gpu_early_stop.py#L115) | `cp.float32` para distâncias e fitness — desalinhado com novo kernel |
| Spec TSPLIB | [second_draft.tex](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex) | Fórmula `nint(d) = floor(d + 0.5)` documentada; pipeline não cumpria |

---

## Impact Analysis

### Root Cause

A função `matrix.py` calculava distâncias corretamente como `int64` internamente (arredondamento TSPLIB), mas as convertia para `float32` na linha 229 antes de retorná-las. `float32` tem apenas 23 bits de mantissa (~7 dígitos decimais). Distâncias individuais cabem nesse limite, mas **o custo total de um tour** (soma de ~$n$ distâncias) ultrapassa $2^{23}$ para instâncias médias/grandes, introduzindo erro de representação silencioso que corrompe o cálculo do gap.

Adicionalmente, o kernel `ga_fujimoto_early_stop.cu` tinha 50 iterações de 2-opt hardcoded, ignorando o parâmetro `two_opt_iterations` da assinatura. Isso criava assimetria paramétrica (FullGPU: 50 iterações vs. Híbrido/CPU: 10), invalidando comparações de iso-algoritmicidade.

### Design Decisions

| Component | Before | After | Rationale |
|---|---|---|---|
| **Distância (matriz)** | `float32` | `int32` | TSPLIB exige inteiros; `int32` cobre até 2.1B (máx. distância terrestre ~20.037) |
| **Fitness/custo total** | `float` (CUDA) / `float32` (python) | `long long` / `int64` | Soma de ~n distâncias pode superar int32; long long cobre até ~9.2×10¹⁸ |
| **Deltas (2-opt)** | `float` | `int` | Diferença de dois inteiros é inteira; `int32` é suficiente |
| **Epsilon 2-opt** | `< current - 1e-5f` | `< current` | Epsilon era workaround para imprecisão float; com int é desnecessário e conceitualmente incorreto |
| **Shared memory** | 4 bytes/thread (`float`) | 8 bytes/thread (`long long`) | Em `cost_calculator.cu`: 256×8=2048 bytes/bloco — desprezível nos 48KB da GTX 1050 |
| **Iterações 2-opt** | `50` hardcoded | `two_opt_iterations` param | Fix de bug: o parâmetro estava sendo ignorado |

---

## Affected Files

### python Wrappers (5 files)

1. [matrix.py L229](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py#L229) — `astype(xp.float32)` → `astype(xp.int32)`
2. [genetic_algorithm_full_gpu_early_stop.py L115](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_full_gpu_early_stop.py#L115) — `cp.float32` → `cp.int32`/`cp.int64`; reordenação de params do kernel
3. [genetic_algorithm_hybrid_naive.py L89](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_hybrid_naive.py#L89) — `cp.float32` → `cp.int32`
4. [genetic_algorithm_hybrid_optimized.py L119](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_hybrid_optimized.py#L119) — `cp.float32` → `cp.int32`/`cp.int64`; shared mem `4` → `8` bytes por thread
5. [genetic_algorithm_cpu.py L119](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/metaheuristics/genetic_algorithm_cpu.py#L119) — `np.float32` → `np.int64`

### CUDA Kernels (5 files)

1. [ga_fujimoto.cu L19](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto.cu#L19) — `float*` dist → `int*`; `float` fitness → `long long`
2. [ga_fujimoto_early_stop.cu L23](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/ga_fujimoto_early_stop.cu#L23) — mesma conversão + `50` hardcoded → `two_opt_iterations` param
3. [two_opt_single.cu L34](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/two_opt_single.cu#L34) — `float *dist` → `int *dist`; deltas `float` → `int`; remoção de epsilon
4. [two_opt_batch.cu L40](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/two_opt_batch.cu#L40) — mesma conversão que `two_opt_single.cu`
5. [cost_calculator.cu L30](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/algorithms/kernels/cost_calculator.cu#L30) — `float *distances` → `int *distances`; acumulação `float` → `long long`

---

## Change Checklist

- [x] 1. `matrix.py` — `float32` → `int32` na saída
- [x] 2. `ga_fujimoto.cu` — `float` → `int`/`long long` nas distâncias e fitness
- [x] 3. `ga_fujimoto_early_stop.cu` — mesma conversão + fix do bug hardcoded 50 iterações
- [x] 4. `two_opt_single.cu` — `float` → `int` em dist/deltas; remoção de epsilon `1e-5f`
- [x] 5. `two_opt_batch.cu` — mesma conversão
- [x] 6. `cost_calculator.cu` — `float*` → `int*` dist; `float` → `long long` custos
- [x] 7. `genetic_algorithm_full_gpu_early_stop.py` — dtypes `cp.float32` → `cp.int32`/`cp.int64`
- [x] 8. `genetic_algorithm_hybrid_naive.py` — `cp.float32` → `cp.int32`
- [x] 9. `genetic_algorithm_hybrid_optimized.py` — `cp.float32` → `cp.int32`/`cp.int64`; shared mem ×2
- [x] 10. `genetic_algorithm_cpu.py` — `np.float32` → `np.int64`

---

## Deep Analysis

> For full arithmetic background on TSPLIB integer arithmetic, see
> [knowledge_tsplib_arithmetic.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/knowledge-base/knowledge_tsplib_arithmetic.md).

### Por que a conversão era necessária

A especificação TSPLIB define que distâncias `EUC_2D` devem ser calculadas como `nint(d) = floor(d + 0.5)`, produzindo **valores inteiros por definição**. O pipeline anterior calculava esses inteiros corretamente em `int64` internamente ([matrix.py L133](file:///home/lucas_galdino/chimera/gpu_accelerated/code/src/distances/matrix.py#L133)), mas **descartava a precisão** ao converter para `float32` na saída. O formato `float32` IEEE 754 possui apenas 23 bits de mantissa (~7 dígitos decimais de precisão), o que significa que o **custo total de um tour** (soma de ~$n$ distâncias inteiras) frequentemente ultrapassa $2^{23}$, corrompendo os gaps de qualidade e impedindo a obtenção de gaps exatos ($0{,}0\%$).

### Prós

1. **Conformidade TSPLIB:** Gaps agora são calculados sobre distâncias exatas, permitindo gaps estritos de $0{,}0\%$ quando o algoritmo encontra o ótimo.
2. **Eliminação de epsilon:** O `1e-5f` existia para mascarar imprecisão de ponto flutuante: dois trajetos matematicamente equivalentes podiam diferir em ~$10^{-6}$ por acumulação de erros, fazendo a condição `new_dist < current_dist` falhar (swap não aplicado) ou passar erroneamente (swap fantasma). Com `int32`, a soma de inteiros é exata — não existe representação intermediária fracionária. O epsilon não só se torna desnecessário como era **conceitualmente incorreto** para dados inteiros: a escolha de `1e-5f` era arbitrária e podia ocultar melhorias reais de 1 unidade de distância inteira.
3. **Reprodutibilidade entre arquiteturas NVIDIA:** CUDA é exclusivamente NVIDIA, portanto "GPUs diferentes" significa diferentes **arquiteturas NVIDIA** (Pascal CC 6.1, Turing CC 7.5, Ampere CC 8.0, etc.). O problema real era que operações `float` podem produzir resultados bit-a-bit diferentes entre arquiteturas porque o padrão IEEE 754 permite — mas não obriga — uso de FMA (Fused Multiply-Add), que reduz erro mas produz resultados distintos de operações separadas. Com `int32`, `a + b` é idêntico em qualquer hardware NVIDIA, garantindo que benchmarks em GTX 1050 e RTX 3090 produzam os **mesmos gaps exatos**.
4. **Throughput de memória inalterado:** `int32` (4 bytes) ocupa o mesmo espaço que `float32`, então a largura de banda de memória global da matriz de distâncias não é afetada.

### Contras

1. **Generalidade perdida fora da TSPLIB:** Dentro da TSPLIB, todos os tipos de aresta (`EUC_2D`, `GEO`, `ATT`, `CEIL_2D`, etc.) já aplicam `nint()` ou `ceil()`, produzindo inteiros — inclusive coordenadas geográficas do tipo `GEO`. A generalidade perdida é para **uso do pipeline fora da TSPLIB**: instâncias proprietárias com distâncias GPS brutas em `float64`, ou instâncias sintéticas com euclideano não-arredondado.
2. **Custo de shared memory em `cost_calculator.cu`:** A acumulação de fitness migrou de `float` (4 bytes) para `long long` (8 bytes). Shared memory é **SRAM on-chip** dentro do SM — separada fisicamente da VRAM — portanto não há impacto nos cálculos de distância nem na VRAM. O custo real é de ocupância: a GTX 1050 possui 48 KB de shared memory por SM (limite Pascal CC 6.1), compartilhados entre todos os blocos ativos no SM. Com `long long`, o `cost_calculator` usa $256 \times 8 = 2048$ bytes por bloco. O trade-off se tornaria **visível** em GPUs mais antigas (Kepler: 16 KB por SM) ou em kernels futuros que também expandam seus buffers.
3. **Risco de overflow em `int32` para distâncias:** Para TSP geográfico, a distância máxima entre dois pontos na Terra é ~20.037 km; o custo total de um tour com $n=10.000$ cidades no pior caso seria ~$2 \times 10^8$ — dentro de `int32` (~$2,1 \times 10^9$), com margem de ~10×. Para **CVRP/MDVRP** (múltiplos veículos, múltiplos depósitos), o custo total (soma de todas as rotas) pode facilmente ultrapassar `int32`, confirmando que `long long` para o **fitness acumulado** é necessário mesmo para problemas de roteamento reais.
4. **Kernel legado não convertido:** O `two_opt_single_cub.cu` (que usa `double`) não foi convertido porque não é referenciado por nenhum wrapper ativo. Serve apenas como referência de padrões de redução CUB.

---

## Additional Findings

> [!WARNING]
> **Bug de iso-algoritmicidade em `ga_fujimoto_early_stop.cu`:** O parâmetro `two_opt_iterations` estava declarado na assinatura da função mas `50` estava hardcoded no corpo do loop — o parâmetro era ignorado. Este bug fazia com que a variante FullGPU executasse 50 iterações de 2-opt por geração enquanto todas as outras variantes executavam 10, invalidando a comparação iso-algorítmica. Corrigido nesta mesma passagem.
