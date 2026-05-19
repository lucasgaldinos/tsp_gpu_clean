---
title: "Knowledge: Ferramentas de Perfilação CUDA"
description: "Comparação entre Nsight Systems, Nsight Compute e CUDA Events (CuPy) para perfilação de kernels GPU. Inclui plano de instrumentação em 3 fases para o projeto Chimera."
created: "2026-05-14 18:30 -0300"
author: "[[Lucas Galdino]]"
tags: [cuda, profiling, nsight, cupy, performance, investigation, kernel]
links:
  - "[plan-roteiro.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/plan-roteiro.md)"
  - "[walkthrough_3.4.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_3.4.md)"
---

# Knowledge: Ferramentas de Perfilação CUDA para o Projeto Chimera


## Objetivo

Consolidar as capacidades, métricas coletáveis, e limitações das ferramentas de perfilação GPU disponíveis para validar as hipóteses arquiteturais dos TODOs pendentes no relatório ([second_draft.tex L2641](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/tex/second_draft.tex#L2641)).

---

## 1. Ferramentas Disponíveis

### 1.1 NVIDIA Nsight Systems (`nsys`)

| Aspecto | Detalhe |
|---|---|
| **Escopo** | Visão **sistêmica** (CPU + GPU + memória + I/O) |
| **Para que serve** | Identificar gargalos macroscópicos: latência de lançamento de kernels, transferências H2D/D2H, sincronizações CPU-GPU |
| **Overhead** | **Baixo** — adequado para traçar execuções longas sem perturbar o comportamento |
| **Saída** | Timeline unificada (formato `.nsys-rep`) visualizável no Nsight Systems GUI |
| **Métricas coletáveis** | Tempo de kernel, duração de `memcpy`, intervalos de ociosidade, concorrência de streams |
| **Limitação** | **Não** fornece métricas internas de kernel (registradores, divergência de warps, ocupação) |

> [!TIP]
> **Usar PRIMEIRO.** O workflow recomendado pela NVIDIA é começar com Nsight Systems para localizar os kernels problemáticos, e só depois mergulhar com Nsight Compute.

**Comando exemplo:**

```bash
nsys profile --trace=cuda,nvtx -o chimera_profile python3 run_benchmark.py
```

---

### 1.2 NVIDIA Nsight Compute (`ncu`)

| Aspecto | Detalhe |
|---|---|
| **Escopo** | Análise **por kernel** individual |
| **Para que serve** | Deep-dive em métricas de SM: eficiência de warps, uso de registradores, shared memory, coalescing de memória global |
| **Overhead** | **Alto** — usa *kernel replay* (re-executa o kernel múltiplas vezes para coletar contadores de hardware) |
| **Saída** | Relatório detalhado (`.ncu-rep`) com *Guided Analysis* |
| **Métricas coletáveis** | |

#### Métricas Relevantes para os TODOs do Relatório

| Hipótese Pendente | Métrica Nsight Compute | Seção no NCU |
|---|---|---|
| Pressão de registradores | `sm__registers_per_thread`, Occupancy Analysis | Speed of Light → SM |
| Divergência de warps | `smsp__warps_issue_stalled_not_selected` / `branch_efficiency` | Warp State Statistics |
| Gargalo compute vs memory | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Speed of Light |
| Coalescing de memória | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_active` | Memory Workload |
| Shared memory bank conflicts | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.avg.pct_of_peak_sustained_active` | Memory Workload |

> [!WARNING]
> **Kernel Replay:** Nsight Compute re-executa cada kernel para coletar diferentes conjuntos de contadores. Isso significa que:
>
> - O tempo total de profiling é **muito** maior que a execução normal
> - Resultados de execuções que dependem de estado (como o GA evolutivo) podem diferir entre replays
> - Não é adequado para medir tempo real; use apenas para métricas estruturais

**Comando exemplo:**

```bash
ncu --set full --kernel-name ga_evolution_early_stop -o chimera_kernel python3 run_benchmark.py
```

---

### 1.3 CUDA Events via CuPy (Instrumentação Não-Intrusiva)

| Aspecto | Detalhe |
|---|---|
| **Escopo** | Temporização precisa de kernels e seções de código no pipeline python |
| **Para que serve** | Medir tempo de kernel, tempo de transferência H2D/D2H, e overhead de lançamento — **sem ferramentas externas** |
| **Overhead** | **Mínimo** — eventos são registrados diretamente no stream CUDA |
| **Limitação** | Fornece **apenas tempo**, sem métricas internas de kernel |

#### Padrão de Uso com CuPy

```python
import cupy as cp

# Criar eventos
start = cp.cuda.Event()
end = cp.cuda.Event()

# Gravar início
start.record()

# Executar kernel
kernel((grid,), (block,), args)

# Gravar fim e sincronizar
end.record()
end.synchronize()

# Calcular tempo em ms
elapsed_ms = cp.cuda.get_elapsed_time(start, end)
```

#### Benchmarking Automatizado (Recomendado)

```python
from cupyx.profiler import benchmark

result = benchmark(minha_funcao, args=(), n_repeat=20)
print(result)  # Imprime média, std, min, max
```

> [!IMPORTANT]
> **Warm-up obrigatório:** A primeira execução de um kernel CuPy RawKernel inclui o tempo de compilação JIT. Sempre executar 2-3 iterações de warm-up antes de coletar dados.

---

## 2. Comparação Consolidada

| Critério | Nsight Systems | Nsight Compute | CUDA Events (CuPy) |
|---|---|---|---|
| **Escopo** | Sistema completo | Kernel individual | Seção de código |
| **Overhead** | Baixo | Alto (replay) | Mínimo |
| **Métricas de tempo** | ✅ | ✅ | ✅ |
| **Métricas de HW** | ❌ | ✅ (registradores, warps, mem) | ❌ |
| **Intrusividade** | Baixa | Alta (altera comportamento) | Nenhuma |
| **Facilidade de uso** | CLI + GUI | CLI + GUI | In-code python |
| **Ideal para** | Etapa 1: localizar gargalos | Etapa 2: diagnosticar causa | Etapa 0: baseline temporal |

---

## 3. Plano de Instrumentação Proposto

### Fase 0: Baseline Temporal (CUDA Events)

**Objetivo:** Coletar tempos precisos de cada fase do pipeline para as 4 variantes (CPU, HybridNaive, HybridOptimized, FullGPU).

**Implementação:**

1. Instrumentar `genetic_algorithm_base.py` com `cp.cuda.Event()` ao redor de cada fase: `_improve_population`, `_evaluate_population`, seleção, crossover, mutação.
2. Para FullGPU (kernel monolítico), medir o tempo total do kernel `ga_evolution_early_stop`.
3. Coletar dados para `kroA100`, `pr1002`, e `pr2392` (representativos de cada faixa de tamanho).

**Entregável:** Tabela de tempos por fase por variante.

---

### Fase 1: Visão Sistêmica (Nsight Systems)

**Objetivo:** Identificar gargalos de transferência e ociosidade no pipeline completo.

**Implementação:**

```bash
nsys profile --trace=cuda,nvtx --output=chimera_phase1 \
    python3 run_chapter4_benchmark.py --instance pr1002 --variant FullGPU
```

**Foco:**

- Confirmar que FullGPU realmente elimina transferências intermediárias
- Medir latência de lançamento do kernel monolítico
- Identificar pontos de sincronização desnecessários

**Entregável:** Screenshots da timeline com anotações.

---

### Fase 2: Diagnóstico de Kernel (Nsight Compute)

**Objetivo:** Validar ou refutar as hipóteses arquiteturais dos TODOs.

**Implementação:**

```bash
ncu --set full \
    --kernel-name ga_evolution_early_stop \
    --launch-count 3 \
    --output=chimera_kernel_analysis \
    python3 run_single_instance.py --instance kroA100
```

**Hipóteses a testar:**

| TODO | Métrica a coletar | Resultado esperado |
|---|---|---|
| Pressão de registradores | `sm__registers_per_thread` | Se > 32, limita ocupação |
| Divergência de warps | `branch_efficiency` | Se < 90%, divergência significativa |
| Compute vs Memory bound | Speed of Light summary | Classificar o kernel |

**Entregável:** Relatório NCU com análise guiada para cada kernel.

---

## 4. Limitações e Cuidados

> [!CAUTION]
> **Replay e GA evolutivo:** O kernel `ga_evolution_early_stop` é um loop de N gerações com estado acumulado. O kernel replay do Nsight Compute pode produzir resultados diferentes entre passes porque o estado interno (população, fitness) evolui. Recomenda-se:
>
> - Limitar o profiling a poucas gerações (`max_generations=5`)
> - Usar `--launch-count 1` para capturar apenas a primeira execução
> - Comparar com dados de CUDA Events para validar consistência temporal

> [!NOTE]
> **Suporte de hardware:** GTX 1050 (Pascal, CC 6.1) suporta todas as métricas básicas do Nsight Compute, mas algumas métricas avançadas (como `warp_stall_reason`) podem ter granularidade limitada comparada a GPUs Turing+ (CC 7.5+).
