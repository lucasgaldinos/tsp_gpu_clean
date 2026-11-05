---
title: "Walkthrough: Task 10.21 — Limpeza de documentação e scripts auxiliares (`DP10-10`)"
description: >
  Histórico de execução da Task 10.21: deleção dos 4 arquivos obsoletos e dos 9 scripts
  one-shot de `routing_data/tmp/`, promoção dos 4 `probe_*.py` a 3 exemplos tipados em
  `routing_data/examples/`, correção de âncoras/links em 13 documentos de `routing_data`,
  deleção de 5 Markdown obsoletos em `gpu_accelerated_clean` e atualização dos caminhos de
  benchmarking.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Registro da execução completa das Seções 5.A–5.D do inventário de documentação,
          com as evidências de `ruff`/`mypy` nos arquivos tocados e do `check_docs` exit 0.
related_files:
  - "[plan-reorganization-and-typing.plan.md](../../documents_tcc/plan/plan-reorganization-and-typing.plan.md)"
  - "[knowledge_tsplib_parser_module_map_and_doc_anchors.md](../../documents_tcc/plan/knowledge-base/knowledge_tsplib_parser_module_map_and_doc_anchors.md)"
  - "[discussion-on-routing_data-gpu_accelerated_clean.md](../../documents_tcc/plan/knowledge-base/discussion-on-routing_data-gpu_accelerated_clean.md)"
tags:
  - notes/walkthrough
  - notes/documentation
  - analysis/architecture
  - guide/etl
  - review/quality
  - guide/testing
  - guide/documentation
  - notes/cleanup
  - analysis/codebase-hygiene
  - notes/maintenance
  - review/code-quality
  - notes/session
---
# Walkthrough: Task 10.21

## 1. Objetivo

Executar o inventário aprovado em `DP10-10`: apagar arquivos de rascunho que já foram
superados pela documentação viva, transformar os `probe_*.py` em exemplos utilizáveis,
sincronizar âncoras `arquivo:linha` e caminhos de pacote com o código real, e manter o
gate `benchmark_scripts.check_docs` em exit 0.

## 2. Correlação de contexto

| Alegação na documentação (antes) | Verdade no código (verificada) |
| --- | --- |
| `models.py` tem 967 linhas com `Field` e `_SectionAccumulator` | `models.py` tem 350 linhas; `Field` em `fields.py`, `_SectionAccumulator` não existe |
| `_preprocess_vrp_file` escreve arquivo temporário removido em `finally` | `parser.py:221` opera só sobre texto em memória |
| `detect_special_distance_type` (`parser.py:854`) | símbolo deletado; nenhum detector de arquivo inteiro |
| `from tsplib_parser.models import FuncT, IntegerField` | `FuncT` em `transformers.py`, `IntegerField` em `fields.py` |
| `pip install -r requirements.txt` | `requirements.txt` não existe; o caminho é `uv sync` |
| testes em `tests/test_converter`, `tests/test_integration` | `tests/converter`, `tests/tsplib_parser`, `tests/e2e` |
| docstrings do converter descrevem a tabela `nodes` v1 | schema v2: `problems`, `tsp_problems`, `cvrp_problems`, `edge_weight_matrices`, … |
| `src/benchmarking_v2/`, `src/benchmarks_v2/`, `code/benchmarks_v2/` | `src/benchmarking/{runner,analysis,database}`, `src/cli/`, `benchmark_scripts/`, `configs/` |
| `GA_PARAMS`, `BENCHMARK_PARAMS` | `configs/algorithms.json`, `configs/benchmark.json` |

## 3. Decisões de projeto

| Decisão | Alternativa descartada | Por quê |
| --- | --- | --- |
| 4 `probe_*.py` → **3** exemplos | manter 4 arquivos | `probe_parse.py` e `probe_parse2.py` eram o mesmo script com dois níveis de tipagem; viraram um único `inspect_parse_result.py` |
| `git rm` para os 17 arquivos de `tmp/` | deixar em `tmp/` | `tmp/` não é versionado como área de rascunho viva; o conteúdo já está superado |
| Exemplos aceitam `--corpus` com default `datasets_raw` | caminho fixo no código | mantém os exemplos utilizáveis fora do repo sem criar dependência de corpus em `src/` |
| Tabela de módulos do `ARCHITECTURE.md` ganhou 3 linhas novas | editar só a linha de `models.py` | o inventário pedia `fields.py` e `tokenization.py` como donos das responsabilidades removidas |
| `reference/README.md` e `PARSER-DETAILS.MD` não foram tocados | corrigir agora | fora do escopo aprovado em 5.B; registrados como achados |

## 4. Log de falhas e correções

| Tentativa | Evidência | Correção |
| --- | --- | --- |
| `inspect_parse_result.py` tipado como `dict[str, Any]` | `mypy`: `Incompatible types in assignment (expression has type "ProblemData")` | usar `ProblemData` de `tsplib_parser.types` |
| imports de `ParseResult`/`ProblemData` em runtime | `ruff TC001` ×3 | movidos para `if TYPE_CHECKING:` |
| primeira edição em `api.py` consumiu o `"""` de fechamento | `ast.parse` falhou com cascata de erros | restaurado `"""` + `# Parse the file` antes de `parsed_data = ...` |

## 5. Arquivos afetados

### `routing_data` — deleções

- `tmp/src_exploration_report.md`, `tmp/analysis-test-suite-organization.md`,
  `tmp/baseline-durations.txt`, `tmp/batch3-durations.txt`.
- 9 scripts one-shot: `add_dataset_fixture.py`, `annotate_tests.py`,
  `fix_cli_fixtures.py`, `fix_imports.py`, `merge_tests.py`, `rewrite_matrix_test.py`,
  `split_cordeau.py`, `tag_tests.py`, `tmp_path_migration.py`.
- 4 `probe_*.py` (substituídos pelos exemplos).

### `routing_data` — novos exemplos

- `examples/inspect_adjacency_edges.py` — fixture inline `ADJ_LIST`, sem corpus.
- `examples/scan_corpus_folders.py` — primeiro arquivo de cada pasta de corpus, inclui Cordeau.
- `examples/inspect_parse_result.py` — contrato de `ParseResult` sobre 4 arquivos.
- `examples/README.md` — índice e comandos.

### `routing_data` — documentos

- `README.md`: badge e contagens 425, `uv sync`, links para `AGENTS.md`, contagens por pasta.
- `docs/README.md`: links de contribuição, seções `src/` e `studies/`.
- `docs/ARCHITECTURE.md`: tabela de módulos (4 linhas), pré-processamento em memória, remoção de `PC6`.
- `docs/src/tsplib_parser/parser-pipeline.md`: âncoras, fluxo sem arquivo temporário, tabela de código vivo.
- `docs/src/tsplib_parser/models.md`: âncoras, imports corretos, `ExtendedProblem`, fim do `Problem.parse`.
- `docs/studies/python-logic/practical/factory-pattern.md`, `practical/fields-vs-dataclasses.md`,
  `theoretical/classes-instances-cls.md`, `theoretical/descriptors-fields.md`.
- `docs/guides/USER_GUIDE.md`: links de contribuição e caminhos de teste.

### `routing_data` — docstrings

- `src/converter/api.py`: payload v2 em `SimpleConverter.parse_file` e `parse_file`.
- `src/converter/output/parquet_writer.py`: exemplo cita as tabelas v2.

### `gpu_accelerated_clean`

- Deletados: `daily_planning.md`, `meeting-notes.md`, `modularizing-cuda-kernels.md`,
  `documentation/analysis/presentations/key_findings.md`,
  `documentation/critical_analysis/criticas_analysis_bib.md`.
- `notes_project_direction_questions.md`: links e notas sobre os dois arquivos absorvidos.
- `README.md`: pacotes `src/benchmarking/` e `src/cli/`, comandos via `benchmark_scripts`.
- `HARDWARE_SPECS.md`: parâmetros apontam para `configs/algorithms.json` e `configs/benchmark.json`.
- `documentation/analysis/benchmarking/benchmarking_system.md`,
  `documentation/analysis/algorithms/module_interactions.md`,
  `documentation/analysis/comparative/algorithm_comparison.md`,
  `documentation/analysis/presentations/fact_check_report.md`.

## 6. Checklist

- [x] 5.A.1–5.A.3: 4 arquivos obsoletos deletados.
- [x] 5.A.4: 9 scripts deletados, 4 probes promovidos a 3 exemplos tipados.
- [x] 5.B: âncoras e links de `routing_data` atualizados.
- [x] 5.C: 5 Markdown deletados sem deixar links quebrados.
- [x] 5.D: caminhos de `gpu_accelerated_clean` atualizados.
- [x] `check_docs` exit 0.

## 7. Evidências de validação

```text
$ .venv/bin/python -m ruff check examples
All checks passed!
$ .venv/bin/python -m ruff format --check examples
4 files already formatted
$ .venv/bin/python -m mypy examples
Success: no issues found in 3 source files
$ .venv/bin/python -m ruff check src/converter/api.py src/converter/output/parquet_writer.py
All checks passed!
$ .venv/bin/python -m mypy src/converter/api.py src/converter/output/parquet_writer.py
Success: no issues found in 2 source files
$ .venv/bin/python examples/inspect_adjacency_edges.py
edge_data_format: ADJ_LIST
top-level edges: [[0, 1], [0, 2], [1, 0], [2, 0]]
$ .venv/bin/python -m pytest tests/tsplib_parser -q        → 171 passed
$ .venv/bin/python -m pytest tests/converter -q            → 244 passed, 1 xfailed
$ .venv/bin/python -m pytest tests/e2e -q                  → 1 passed
$ .venv/bin/python -m pytest tests/test_unit_conftest.py -q → 9 passed
$ .venv/bin/python -m benchmark_scripts.check_docs
stale docstrings: 0
dead links: 0
undefined symbols: 0
OK: no documentation drift detected   (exit 0)
```

Observação: `scan_corpus_folders.py` relata `FAIL ValidationError` para
`cvrplib/Holmberger` (`C1_10_1.txt`) e `cvrplib/Vrp-Set-Solomon` (`C101.txt`) — são
formatos de terceiros (Solomon/Homberger), não TSPLIB95, e o exemplo imprime a falha em
vez de mascará-la. O comportamento é o esperado.

## 8. Achados incidentais (não corrigidos, fora de escopo)

| Achado | Evidência | Próximo passo |
| --- | --- | --- |
| `docs/ARCHITECTURE.md` §9 ainda descreve `tests/test_format/`, `tests/test_converter/`, `tests/test_integration/` e "14 failures" | `docs/ARCHITECTURE.md:490-520` | registrar como tarefa própria de atualização da estratégia de testes |
| `docs/reference/README.md` linka `CONTRIBUTING.md` inexistente | `docs/reference/README.md:17,51` | redirecionar para `AGENTS.md` |
| `parser-pipeline.md` e `models.md` linkam `src/tsplib_parser/PARSER-DETAILS.MD`, ausente | frontmatter `related_files` | decidir entre restaurar o arquivo ou remover os links |
| `documents_tcc/how-to-tcc.md` ainda cita `benchmarks_v2`/`src/benchmarking_v2` | `how-to-tcc.md:184-188,325-341` | fora do escopo de 5.D |
| `README.md` de `routing_data` ainda mostra `-t nodes` e a tabela `nodes` no schema | `README.md:110-160` | decidir se o schema v2 substitui a seção inteira |

## 9. Análise profunda

O custo real desta tarefa não foi deletar arquivos, e sim **decidir o que é fonte de
verdade**. `routing_data/docs/studies/python-logic/**` é uma série didática que cita
linhas exatas de `models.py`; qualquer decomposição de módulo invalida dezenas de
âncoras. A regra adotada — citar o símbolo e depois a linha — mantém o texto legível
quando a linha muda, porque o leitor ainda consegue localizar o `def`.

A promoção dos probes também mudou o contrato dos arquivos: o que era `print` de
depuração virou função tipada com `ParseResult`/`ProblemData`, sob `mypy` strict e
`ruff`. Isso só foi possível porque a Task 10.x anterior tipou o retorno do parser; é o
primeiro consumidor externo que se beneficia daquele contrato.
