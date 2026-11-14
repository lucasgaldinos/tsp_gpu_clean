---
# --- Antigravity rule keys ---
trigger: glob
globs: "documents_tcc/**/*.md, documents_tcc/**/*.tex, documentation/walkthroughs/**/*.md" # shared with VS Code
description: "Use when writing Portuguese prose in plans, walkthroughs, or TCC chapters: Brazilian register, anti-calque rules, and the artifact language map."
# --- VS Code instruction keys ---
name: "Prose Language"
applyTo: "documents_tcc/**/*.md, documents_tcc/**/*.tex, documentation/walkthroughs/**/*.md"
---

# Prose language — pt-BR reference

Single source of truth for which language an artifact uses, and for how Portuguese is written when translation is unavoidable. Agent-facing skills stay in English; this file supplies the mapping they must follow when their output lands in a Portuguese artifact.

## Language map

| Artifact | Language |
| --- | --- |
| `documents_tcc/plan/**`, `documentation/walkthroughs/**` | pt-BR |
| `documents_tcc/tex/**`, `documents_tcc/*.tex` | pt-BR (ABNT register) |
| `AGENTS.md`, `.agents/**`, `.github/**` | English |
| `documentation/knowledge-base/**`, `documentation/analysis/**` | English |
| Code identifiers, docstrings, comments | English |
| Commit subjects | English type/scope prefix; free text may be pt-BR |

One document, one language. Never mix translated headings into an English document or English headings into a **`pt-BR`** document, except for the canonical technical terms listed below.

## Brazilian register, not a calque

The failure mode to avoid is a word-for-word English calque that reads like European Portuguese. Write the sentence a Brazilian engineer would say, then keep the technical term.

| Calque / pt-PT | pt-BR |
| --- | --- |
| ficheiros | arquivos |
| ecrã | tela |
| equipa | equipe |
| registo | registro |
| planeamento | planejamento |
| utilizador | usuário |
| está a executar | está executando |
| está a ser validado | está sendo validado |
| ligação | conexão (rede) / vínculo (conceitual) |
| em baixo | abaixo |
| de forma a | para |
| ao nível de | no nível de |
| veio a permitir | permitiu |
| faz uso de | usa |

Heading calques are the most visible symptom:

| English heading | Wrong (literal) | pt-BR |
| --- | --- | --- |
| DoD Compliance | Atendimento ao DoD | DoD atendido |
| Changed Files | Arquivos Alterados | Arquivos alterados |
| Summary | Resumo | Resumo |
| Technical Justification | Justificativa Técnica | Justificativa técnica |
| Artifacts | Artefatos | Artefatos |
| Next steps | Próximos passos | Próximos passos |
| Walkthrough | Passo a passo | Walkthrough (keep) |
| Trade-offs | Compensações | Trade-offs (keep) |

## Technical terms stay verbatim

Keep the domain's canonical token in English and inflect only the surrounding article or adjective: *o kernel*, *a GPU*, *os checkpoints*, *a thread*, *o grid*, *a baseline*, *o speedup*, *a occupancy*, *o commit*, *a branch*, *o merge*, *o rollback*, *a gate*.

Never translate or reformat:

- Identifiers, paths, commands, flags, versions, log excerpts, and error strings.
- Numbers inside code blocks, copied tables, or log lines. In prose, use the Brazilian decimal comma (`26,27×`).
- Paper titles and quoted passages.

## When the source is an English template

Translate the intent, not the words. If the template heading has no natural pt-BR equivalent, keep the English term in code formatting and explain it once at first use. A heading that only makes sense as a translation ("Atendimento ao DoD") is a defect, not a style preference.

## Related

- Anchor grammar: [markdown.instructions.md](./markdown.instructions.md#referencing)
- Plan and Solution block format: [update-plan.instructions.md](./update-plan.instructions.md)
