---
title: "Knowledge: Aritmética TSPLIB e Tipos Inteiros em Pipelines GPU"
description: "Fundamentação sobre a especificação TSPLIB de distâncias inteiras, implicações de float32 em acumulações de custo, trade-offs de int32 vs long long em kernels CUDA, e análise de overflow para TSP e variantes (VRP, CVRP, MDVRP)."
created: "2026-05-17 21:00 -0300"
author: "[[Lucas Galdino]]"
tags: [tsplib, cuda, arithmetic, integer, float32, overflow, precision, kernel, pipeline]
links:
  - "[plan-roteiro.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/plan-roteiro.md)"
  - "[walkthrough_3.1.md](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_3.1.md)"
---

# Knowledge: Aritmética TSPLIB e Tipos Inteiros em Pipelines GPU

## Objective

Documentar a especificação de distâncias inteiras da TSPLIB, as implicações práticas de usar `float32` em acumulações de custo de tour, e os trade-offs de escolha de tipos (`int32`, `long long`) em kernels CUDA e wrappers Python. Útil para qualquer tarefa que toque em cálculo de distâncias, fitness, ou instrumentação de novos kernels no projeto.

---

## 1. A Especificação TSPLIB

Todas as instâncias TSPLIB com `EDGE_WEIGHT_TYPE` definido aplicam arredondamento, produzindo **distâncias inteiras por definição**:

| Tipo | Fórmula | Resultado |
|---|---|---|
| `EUC_2D` | `nint(d) = floor(d + 0.5)` | Inteiro |
| `CEIL_2D` | `ceil(d)` | Inteiro |
| `ATT` | `ceil(sqrt(...))` | Inteiro |
| `GEO` | Haversine com `nint()` | Inteiro |
| `EXPLICIT` | Fornecida diretamente | Inteiro (declarado na matriz) |

> [!IMPORTANT]
> Mesmo `GEO` (coordenadas geográficas) produz inteiros — a fórmula de haversine é aplicada com arredondamento. A "perda de generalidade" ao adotar `int32` refere-se exclusivamente a **instâncias fora da TSPLIB** (GPS bruto em `float64`, instâncias sintéticas sem arredondamento).

**Referência:** Reinelt, G. (1991). *TSPLIB — A Traveling Salesman Problem Library*. ORSA Journal on Computing.

---

## 2. Por que `float32` Corrompe Acumulações de Custo

`float32` (IEEE 754 single precision) tem **23 bits de mantissa**, correspondendo a ~7 dígitos decimais de precisão. O problema não ocorre em distâncias individuais (cada distância TSPLIB raramente passa de $10^5$), mas sim no **custo total do tour**:

```
Custo total = sum(dist[i][j] for all edges in tour)
            = ~n * avg_dist
```

Para instâncias médias (`n = 1000`, `avg_dist ≈ 300`):
```
Custo ≈ 300.000 > 2^18 = 262.144
```

Para instâncias grandes (`n = 10.000`, `avg_dist ≈ 1.000`):
```
Custo ≈ 10.000.000 > 2^23 = 8.388.608  ← limite de representação exata do float32
```

Acima de $2^{23}$, `float32` representa apenas múltiplos de 2, depois múltiplos de 4, etc. Isso significa que **dois tours distintos podem ter o mesmo custo representado em `float32`**, invalidando comparações e impedindo gaps exatos de $0{,}0\%$.

> [!WARNING]
> O erro não é detectável por testes simples — o pipeline "funciona" e produz resultados plausíveis, mas os gaps calculados são ligeiramente incorretos de forma silenciosa.

---

## 3. Escolha de Tipos: Guia de Referência

### 3.1 Distâncias Individuais

| Tipo | Bytes | Max valor | Adequado para TSPLIB? |
|---|---|---|---|
| `float32` | 4 | ~$10^7$ exato | ❌ — erro de representação em custos acumulados |
| `int32` | 4 | ~$2{,}1 \times 10^9$ | ✅ — distâncias individuais cabem com larga margem |
| `int64` / `long long` | 8 | ~$9{,}2 \times 10^{18}$ | ✅ mas desnecessário para distâncias individuais |

**Decisão para este projeto:** `int32` para a matriz de distâncias. Mesma largura de banda que `float32` (4 bytes), sem custo adicional de memória global.

### 3.2 Fitness / Custo Total do Tour

| Problema | Custo típico máximo | Tipo necessário |
|---|---|---|
| TSP (`n ≤ 1.000`, coords terrestres) | ~$10^6$ | `int32` seria suficiente |
| TSP (`n = 10.000`, coords terrestres) | ~$2 \times 10^8$ | `int32` (margem ~10×) |
| TSP geográfico (distância máx. Terra: 20.037 km) | ~$2 \times 10^8$ para tour completo | `int32` (margem ~10×) |
| CVRP/MDVRP ($k$ veículos) | $k \times$ custo TSP | ⚠️ pode ultrapassar `int32` para $k > 10$ |
| Coordenadas sintéticas não-geográficas | Indefinido | `int64` por precaução |

**Decisão para este projeto:** `long long` / `int64` para fitness acumulado. Cobre todos os cenários sem risco.

> [!TIP]
> A regra prática: **distâncias individuais em `int32`**, **somas de distâncias em `long long`**. Isso é análogo ao padrão da stdlib C de usar `size_t` para contagens e `ptrdiff_t` para diferenças.

---

## 4. Implicações em Kernels CUDA

### 4.1 Shared Memory

A escolha de `long long` para acumulação de fitness **dobra** o consumo de shared memory em kernels de redução:

```
float    → 4 bytes/thread → 256 threads × 4 = 1.024 bytes/bloco
long long → 8 bytes/thread → 256 threads × 8 = 2.048 bytes/bloco
```

> [!NOTE]
> Shared memory é **SRAM on-chip** dentro do SM — fisicamente separada da VRAM (GDDR5/GDDR6 off-chip). Aumentar shared memory **não** impacta a largura de banda de memória global nem o consumo de VRAM.

O impacto real é de **ocupância**: o SM compartilha os 48 KB (Pascal CC 6.1) entre todos os blocos ativos. Com `long long`, cada bloco usa 2 KB em vez de 1 KB. Isso se torna visível quando muitos blocos competem pelo mesmo SM simultaneamente, ou em GPUs mais antigas com menos shared memory por SM:

| Arquitetura | SM Generation | Shared mem/SM | Impacto de 2KB/bloco |
|---|---|---|---|
| Fermi (CC 2.x) | 2010 | 48 KB | Max 24 blocos → 48 KB usados |
| Kepler (CC 3.x) | 2012 | 16–48 KB | Max 8 blocos com 16 KB |
| Maxwell (CC 5.x) | 2014 | 48–96 KB | Irrelevante |
| Pascal (CC 6.x) — GTX 1050 | 2016 | 48 KB | Max 24 blocos → OK |
| Turing+ (CC 7.5+) | 2018+ | 64–96 KB | Irrelevante |

### 4.2 Comparações no 2-opt

Com `float`, era necessário um epsilon de tolerância na comparação de melhoria:

```c
// Antes (float): necessário para evitar "swaps fantasma" por imprecisão
if (new_dist < current_dist - 1e-5f) { ... }

// Depois (int): comparação exata — inteiros não têm representação intermediária
if (new_dist < current_dist) { ... }
```

A escolha de `1e-5f` era arbitrária. Um epsilon maior ocultaria melhorias reais de 1 unidade; um epsilon menor não compensaria o erro de arredondamento em todos os casos. Com `int32`, o problema desaparece — a comparação é sempre exata.

### 4.3 Reprodutibilidade entre Arquiteturas NVIDIA

Operações `float` podem produzir resultados bit-a-bit diferentes entre arquiteturas NVIDIA porque o padrão IEEE 754 **permite mas não obriga** FMA (Fused Multiply-Add). GPUs que usam FMA acumulam menos erro, mas produzem resultados numericamente distintos de operações separadas.

Com `int32`:
- `a + b` é idêntico em Pascal, Turing, Ampere — sem exceção
- Benchmarks em GTX 1050 e RTX 3090 produzem os **mesmos gaps exatos**
- Elimina um fator de confusão em comparações cross-hardware

> [!NOTE]
> CUDA é exclusivamente NVIDIA — "reprodutibilidade entre GPUs" significa entre **arquiteturas NVIDIA**, não entre fabricantes.

---

## 5. Fora do Escopo TSPLIB

Se o pipeline for adaptado no futuro para instâncias **fora da TSPLIB**, considerar:

| Caso | Recomendação |
|---|---|
| Coordenadas GPS brutas (`float64`) | `float64`/`double` para distâncias; `int64` para custos se arredondamento for aplicado |
| Instâncias sintéticas sem arredondamento | `double` para distâncias; avaliar se arredondamento faz sentido para o problema |
| Grafos de roteamento com pesos fracionários | Multiplicar por fator de escala (ex: ×1000) e usar `int64` |

O padrão de "multiplicar por fator de escala" é comum em solucionadores de otimização exata (ex: Concorde) precisamente para evitar aritmética de ponto flutuante.

---

## References / Sources

- Reinelt, G. (1991). *TSPLIB — A Traveling Salesman Problem Library*. ORSA Journal on Computing, 3(4), 376–384.
- IEEE 754-2008 Standard for Floating-Point Arithmetic — especificação de `binary32` (float32).
- NVIDIA CUDA C++ Programming Guide — Shared Memory, SM Architecture (Pascal CC 6.1).
- [walkthrough_3.1.md — Deep Analysis](file:///home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/plan/task-resolutions/walkthrough_3.1.md) — contexto de aplicação no projeto Chimera.
