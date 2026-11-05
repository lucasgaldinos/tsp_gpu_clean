---
title: "Knowledge: mapa de módulos do `tsplib_parser` e âncoras de documentação"
description: >
  Referência reutilizável do estado pós-decomposição (PP1) do pacote `tsplib_parser`:
  qual módulo possui cada responsabilidade, o `parse` em dois passes, o pré-processamento
  VRP puramente em memória, as tabelas do schema v2 e a regra de como escrever âncoras
  `arquivo:linha` que não envelhecem em uma sprint.
created: 2026-11-03
modifications:
  - date_modified: 2026-11-03
    modifications:
      - description: >
          Extraído da execução da Task 10.21: mapa de módulos verificado em código,
          contagens de linhas medidas, e a lista de referências proibidas
          (`_SectionAccumulator`, `detect_special_distance_type`, arquivos temporários,
          tabela `nodes`).
related_files:
  - "[walkthrough-10.21-docs-and-scripts-cleanup.md](../../../documentation/walkthroughs/walkthrough-10.21-docs-and-scripts-cleanup.md)"
  - "[plan-reorganization-and-typing.plan.md](../plan-reorganization-and-typing.plan.md)"
tags:
  - analysis/architecture
  - guide/etl
  - guide/tsplib-parser
  - notes/reference
  - review/quality
  - notes/documentation
---
# Mapa de módulos do `tsplib_parser` (pós-PP1)

Fonte: `/home/lucas_galdino/chimera/routing_data/src/tsplib_parser/`.

| Módulo | Linhas | Possui | Símbolos âncora |
| --- | --- | --- | --- |
| `types.py` | 514 | vocabulário da spec, gramáticas de seção, `ParseResult`/`ProblemData` | `ParseResult` `:374`, `ProblemData` `:292` |
| `exceptions.py` | 129 | hierarquia de exceções | `FormatError` `:24` |
| `matrix.py` | 363 | classes de matriz `EXPLICIT` + registro de formatos | `Matrix` `:54`, `TYPES` `:353` |
| `tokenization.py` | 100 | split lexical em `(headers, sections)` | `_split_tsplib_text` `:45` |
| `transformers.py` | 388 | primitivas de desserialização de texto | `BiSep` `:17`, `FuncT` `:87`, `EdgeDataT` `:127` |
| `fields.py` | 270 | descritores `Field` (uma classe por keyword/seção) | `Field` `:24`, `IntegerField` `:92`, `parse_for_problem` `:62`/`:203` |
| `models.py` | 350 | hierarquia de modelos + registro de problemas | `FileMeta` `:33`, `Problem` `:57`, `StandardProblem` `:87`, `ExtendedProblem` `:236`, `PROBLEM_REGISTRY` `:288` |
| `parser.py` | 585 | entrada ETL: preprocess → parse → validate → extract | `FormatParser` `:76`, `parse_file` `:105`, `_preprocess_vrp_file` `:221`, `validate_problem` `:569` |
| `validation.py` | 645 | invariantes semânticas sobre dicts | `check_symmetry` `:124`, `validate_problem_data` `:191`, `validate_parse_result` `:412` |

## `StandardProblem.parse` em dois passes (`models.py:121`)

```text
problem = cls()                                        :137
raw_headers, raw_sections = _split_tsplib_text(...)    :139
for header in raw_headers:  field.parse(value)         :141-147
for section in raw_sections: sec_field.parse_for_problem(body, problem)  :151-159
if errors: raise ParseError(...)                       :164
return problem                                         :165
```

Consequências para a documentação:

- Não existe `_SectionAccumulator`: o corpo de cada seção é entregue inteiro a
  `Field.parse_for_problem`, que pode ler headers já populados (`EDGE_DATA_FORMAT`).
- `Problem` não declara `parse` (correção de `PB6`); não há caminho "parse silencioso
  devolve instância vazia".

## Pré-processamento VRP é puramente em memória

`_preprocess_vrp_file` (`parser.py:221`) recebe o texto já lido, remove as seções
estendidas e devolve `(text, additional_fields)`. Nenhum arquivo temporário é escrito,
portanto qualquer menção a `tempfile`/`finally: unlink` na documentação está obsoleta.

## Schema v2 nas docstrings do converter

`nodes.parquet` e a tabela `nodes` v1 não existem. O conjunto exportável é
`ALLOWED_TABLES` em `src/converter/output/parquet_writer.py`: `problems`,
`tsp_problems`, `atsp_problems`, `cvrp_problems`, `hcp_problems`, `sop_problems`,
`edge_weight_matrices`, `file_tracking`, `solutions`.

## Regra de âncora que não envelhece

1. Cite o **símbolo** primeiro (`StandardProblem.parse`) e a linha entre parênteses —
   quando a linha envelhecer, o leitor ainda acha o símbolo.
2. Prefira o início do `def`/`class` em vez de uma linha interna do corpo.
3. Ao decompor um módulo, atualize: mapa de módulos, diagramas de camadas, tabelas de
   símbolos e as séries `studies/python-logic/**` no mesmo commit.
4. Nunca nomeie um arquivo que o código já não escreve (`tempfile`, `requirements.txt`)
   nem um símbolo deletado (`detect_special_distance_type`).
