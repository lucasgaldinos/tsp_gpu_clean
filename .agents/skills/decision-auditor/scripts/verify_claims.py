#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["google-genai>=3.2.0"]
# ///
"""Evaluate many atomic questions over shared file context with `gemini-decision-1`.

The Decisions API returns calibrated probabilities with zero output tokens, so a
500-rule audit costs only input tokens and reports the exact symbols that failed
instead of prose the caller has to parse.

Two capabilities matter here:

* **AST grounding.** Python sources are parsed locally and a compact structural
  index (class/function/method/closure, line range, parameters, branch count,
  call targets) is prepended to the shared input. The calling agent never has to
  read or emit the AST, so no context is spent on it.
* **Per-symbol localization.** The API never shows question keys to the model, so
  targeting must live in `instructions`. A question marked `"per_symbol": true`
  is cloned once per discovered symbol with an explicit target prefix; the model
  then answers that one question about that one symbol, and the key that comes
  back names the failing function and its line range.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

MODEL = "gemini-decision-1"
MAX_QUESTIONS_PER_REQUEST = 1_000
MAX_CONTEXT_TOKENS = 262_144
QUESTION_KEY_LIMIT = 64
INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_-]")

NOUL_FLAG_THRESHOLD = 0.80
NOUL_CLEAN_THRESHOLD = 0.20
CHOICE_CONFIDENCE_FLOOR = 0.75
SCORE_CONFIDENCE_FLOOR = 0.70
SIGNAL_SPREAD = 0.30
PAIRWISE_PHRASES = (
    "another symbol",
    "another function",
    "another method",
    "other symbol",
    "other function",
    "other method",
    "elsewhere in the file",
    "elsewhere in the module",
    "similar to another",
)
NEGATION_PATTERN = re.compile(
    r"\b(does not|doesn't|is not|isn't|never|fails to|not)\b", re.IGNORECASE
)
BENIGN_NEGATION = re.compile(r"\b(whether or not|if not|or not|not only)\b", re.IGNORECASE)
SELF_REFERENCE_PATTERN = re.compile(
    r"\b(this symbol|this function|this method)\b", re.IGNORECASE
)
MAX_REPORT_ITEMS = 200
CHARS_PER_TOKEN = 4  # optimistic ratio (code tokenises denser than prose)
CONSERVATIVE_CHARS_PER_TOKEN = 3  # measured lower bound; use this for budget caps

CALLABLE_KINDS = {"function", "method", "nested"}
ALLOWED_QUESTION_FIELDS = {"type", "instructions", "criteria"}
ALLOWED_QUESTION_TYPES = {"noul", "choice", "score"}
PLACEHOLDER_KEYS = {"", "placeholder", "your-key-here", "changeme"}


@dataclass(frozen=True)
class Symbol:
    """One class, function, method, or nested closure discovered by the AST."""

    module: str
    qualname: str
    lineno: int
    end_lineno: int
    kind: str
    args: tuple[str, ...] = ()
    if_branches: int = 0
    calls: tuple[str, ...] = ()
    shape: str = ""
    fingerprint: str = ""
    canary: bool = False

    @property
    def label(self) -> str:
        return f"{self.qualname} in {self.module} (L{self.lineno}-L{self.end_lineno})"

    @property
    def raw_id(self) -> str:
        """Stable human-readable identifier; sanitized only when used as a key."""
        return f"{self.module}::{self.qualname}::L{self.lineno}-L{self.end_lineno}"


def _param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    params = node.args
    return tuple(p.arg for p in (*params.posonlyargs, *params.args, *params.kwonlyargs))


def _branch_count(node: ast.AST) -> int:
    return sum(1 for child in ast.walk(node) if isinstance(child, (ast.If, ast.IfExp)))


def _unparse_target(node: ast.expr) -> str | None:
    """Return a call target as text, or None when the node cannot be unparsed."""
    try:
        return ast.unparse(node)
    except (AttributeError, RecursionError, TypeError, ValueError):
        return None


def _call_targets(node: ast.AST, limit: int = 12) -> tuple[str, ...]:
    targets: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = _unparse_target(child.func)
            if target is not None:
                targets.add(target)
    return tuple(sorted(targets)[:limit])


_IGNORED_SHAPE_FIELDS = frozenset({"annotation", "returns", "type_comment", "ctx", "type_params"})


def _shape(node: object) -> str:
    """Serialize an AST node with identifiers erased, keeping structure and literals."""
    if isinstance(node, ast.Constant):
        return f"Const({node.value!r})"
    if isinstance(node, ast.AST):
        fields = ",".join(
            f"{field}={_shape(value)}"
            for field, value in ast.iter_fields(node)
            if field not in _IGNORED_SHAPE_FIELDS
        )
        return f"{type(node).__name__}({fields})"
    if isinstance(node, list):
        return f"[{','.join(_shape(item) for item in node)}]"
    if isinstance(node, str):
        return "_"
    return repr(node)


def _structural_fingerprint(node: ast.AST) -> str:
    """Hash of the identifier-erased body; equal hashes mark duplication candidates."""
    body = getattr(node, "body", node)
    return hashlib.blake2s(_shape(body).encode("utf-8"), digest_size=8).hexdigest()


def _body_shape(node: ast.AST) -> str:
    """The identifier-erased body text that the fingerprint hashes."""
    return _shape(getattr(node, "body", node))


_SHAPE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|Const\([^)]*\)")


def _shingles(shape: str, width: int = 5) -> frozenset[str]:
    """Structural 5-grams, used to catch near-duplicates that exact hashing misses."""
    tokens = _SHAPE_TOKEN.findall(shape)
    if len(tokens) < width:
        return frozenset(tokens)
    return frozenset(
        " ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)
    )


class _SymbolVisitor(ast.NodeVisitor):
    """Collect every class and callable, including closures defined inside callables."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._stack: list[tuple[str, str]] = []
        self.symbols: list[Symbol] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, "class")
        self._stack.append((node.name, "class"))
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_callable(node)

    def _visit_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not self._stack:
            kind = "function"
        elif self._stack[-1][1] == "class":
            kind = "method"
        else:
            kind = "nested"
        self._record(node, kind)
        self._stack.append((node.name, kind))
        self.generic_visit(node)
        self._stack.pop()

    def _record(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        qualname = ".".join([*(name for name, _ in self._stack), node.name])
        self.symbols.append(
            Symbol(
                module=self.module,
                qualname=qualname,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None) or node.lineno,
                kind=kind,
                args=(
                    _param_names(node)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else ()
                ),
                if_branches=_branch_count(node),
                calls=_call_targets(node),
                shape=_body_shape(node),
                fingerprint=_structural_fingerprint(node),
            )
        )


def extract_symbols(path: Path, source: str) -> list[Symbol]:
    """Parse `source` and return its symbols; a syntax error is a warning, not a crash."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(f"warning: no AST index for {path}: {exc}", file=sys.stderr)
        return []
    visitor = _SymbolVisitor(path.stem)
    visitor.visit(tree)
    return visitor.symbols


def render_ast_index(path: Path, symbols: Iterable[Symbol]) -> str:
    """Compact structural summary prepended to a Python file's source block."""
    lines = [f"--- AST STRUCTURAL INDEX FOR {path} ---"]
    for symbol in symbols:
        if symbol.kind == "class":
            lines.append(f"class `{symbol.qualname}` (L{symbol.lineno}-L{symbol.end_lineno})")
            continue
        lines.append(
            f"{symbol.kind} `{symbol.qualname}` (L{symbol.lineno}-L{symbol.end_lineno}) "
            f"args={list(symbol.args)} if_branches={symbol.if_branches} calls={list(symbol.calls)}"
        )
    return "\n".join(lines)


def read_text_file(path: Path) -> str:
    """Read text, refusing binaries so token budget is never spent on garbage."""
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise SystemExit(
            f"Error: {path} looks binary. Pass PDFs as a document block instead of a text file."
        )
    return data.decode("utf-8", errors="replace")


def extract_file_symbols(path: Path, content: str | None = None) -> list[Symbol]:
    """Symbols declared by one file; non-Python files declare none.

    `content` is accepted so a caller that already read the file does not read it twice.
    """
    if path.suffix != ".py":
        return []
    return extract_symbols(path, content if content is not None else read_text_file(path))


def build_shared_context(
    file_paths: list[Path], include_ast: bool = True
) -> tuple[str, list[Symbol]]:
    """Assemble every file plus its AST index into the single shared input payload."""
    chunks: list[str] = []
    symbols: list[Symbol] = []
    for path in file_paths:
        content = read_text_file(path)
        file_symbols = extract_file_symbols(path, content)
        symbols.extend(file_symbols)
        if include_ast and file_symbols:
            chunks.append(render_ast_index(path, file_symbols))
        chunks.append(f"=== FILE: {path} ===\n{content}\n=== END FILE: {path} ===")
    return "\n\n".join(chunks), symbols


def collect_symbols(file_paths: list[Path]) -> list[Symbol]:
    """Discover symbols without building or tokenising the shared context payload."""
    return [symbol for path in file_paths for symbol in extract_file_symbols(path)]


def candidate_clusters(
    symbols: list[Symbol],
    include_classes: bool = False,
    similarity: float = 0.60,
    canary_stems: set[str] | None = None,
) -> dict[str, Any]:
    """Propose duplication candidates: exact shape matches plus near-duplicate pairs.

    Detection is deliberately handled here, in code, so the model is only asked the
    part it is good at: whether a given pair is *accidental* duplication or a shape a
    protocol requires. Over-generating candidates is acceptable; each one costs a
    single question, and the model filters them.
    """
    targets = symbols if include_classes else [s for s in symbols if s.kind in CALLABLE_KINDS]
    stems = canary_stems or set()
    groups: dict[str, list[Symbol]] = {}
    for symbol in targets:
        if symbol.fingerprint:
            groups.setdefault(symbol.fingerprint, []).append(symbol)

    def describe(group: list[Symbol]) -> list[dict[str, Any]]:
        return [
            {
                "id": symbol.raw_id,
                "label": symbol.label,
                "kind": symbol.kind,
                "canary": symbol.module in stems,
            }
            for symbol in group
        ]

    def from_canary(group: list[Symbol]) -> bool:
        return any(symbol.module in stems for symbol in group)

    clusters: list[dict[str, Any]] = [
        {
            "kind": "exact",
            "score": 1.0,
            "size": len(group),
            "canary": from_canary(group),
            "symbols": describe(group),
        }
        for group in groups.values()
        if len(group) > 1
    ]

    shingled = [(symbol, _shingles(symbol.shape)) for symbol in targets if symbol.shape]
    for left in range(len(shingled)):
        for right in range(left + 1, len(shingled)):
            left_symbol, left_set = shingled[left]
            right_symbol, right_set = shingled[right]
            if left_symbol.fingerprint == right_symbol.fingerprint:
                continue
            if not left_set or not right_set:
                continue
            union = left_set | right_set
            score = len(left_set & right_set) / len(union)
            if score >= similarity:
                clusters.append(
                    {
                        "kind": "near",
                        "score": round(score, 3),
                        "size": 2,
                        "canary": from_canary([left_symbol, right_symbol]),
                        "symbols": describe([left_symbol, right_symbol]),
                    }
                )

    clusters.sort(key=lambda cluster: (-cluster["score"], -cluster["size"]))
    return {
        "candidates": len(clusters),
        "exact": sum(1 for cluster in clusters if cluster["kind"] == "exact"),
        "near": sum(1 for cluster in clusters if cluster["kind"] == "near"),
        "similarity_threshold": similarity,
        "symbols_in_candidates": sum(cluster["size"] for cluster in clusters),
        "clusters": clusters,
    }


def cluster_questions(clusters: list[dict[str, Any]]) -> dict[str, Any]:
    """Two independent calibrated yes/no questions per candidate.

    Measured on two known duplicates and two known non-duplicates, the yes/no pair separated
    them with a 0.32 margin, against 0.23 for a three-way choice and 0.18 for an ordered
    score. The choice format also tripled the reading on the known non-duplicate pair
    (0.15 -> 0.47), because choice probabilities sum to one across the options even when none
    of them fits. Independent questions do not have that floor: "neither" is expressible as
    two low probabilities, which a single forced choice cannot say.
    """
    questions: dict[str, Any] = {}
    for index, cluster in enumerate(clusters):
        labels = ", ".join(f"`{symbol['label']}`" for symbol in cluster["symbols"])
        relation = (
            "an identical identifier-erased AST shape"
            if cluster["kind"] == "exact"
            else f"a near-identical identifier-erased AST shape (similarity {cluster['score']})"
        )
        common: dict[str, Any] = {"canary": True} if cluster.get("canary") else {}
        questions[make_question_key("accidental_duplication", f"candidate{index}")] = {
            "type": "noul",
            "instructions": (
                f"The input contains these symbols, which share {relation}: {labels}. "
                "Is that shared shape accidental duplication that should be collapsed into a "
                "single shared helper?"
            ),
            "criteria": {
                "true_description": (
                    "The same logic was written twice, so one symbol could call the other or both "
                    "could delegate to a shared helper."
                ),
                "false_description": (
                    "The shapes coincide for a reason: different semantics, a method a protocol "
                    "requires, or plain coincidence."
                ),
            },
            "flag_at": 0.30,
            "basis": (
                "A/B on 2 known duplicates (0.47, 0.97) and 2 known non-duplicates (0.03, 0.15) "
                "puts the separating window at (0.15, 0.47]; 0.30 sits inside it. Revisit as the "
                "labeled set grows."
            ),
            **common,
        }
        questions[make_question_key("protocol_required", f"candidate{index}")] = {
            "type": "noul",
            "instructions": (
                f"The input contains these symbols, which share {relation}: {labels}. "
                "Is that identical shape required by a protocol or by the language, such that "
                "collapsing the two into one helper would break required behaviour?"
            ),
            "criteria": {
                "true_description": (
                    "A framework dispatches by name, or the language requires a distinct override "
                    "or signature, so both symbols must exist."
                ),
                "false_description": "Nothing requires two separate symbols with this shape.",
            },
            "informational": True,
            **common,
        }
    return questions


def make_question_key(prefix: str, suffix: str = "") -> str:
    """Build an API-legal key (`[A-Za-z0-9_-]+`, <= 64 chars) without collisions.

    Truncation alone is unsafe: two symbols sharing a long prefix would collapse
    onto one key and one answer would silently overwrite the other. A digest of
    the *full* identifier keeps the key unique after shortening.
    """
    raw = prefix if not suffix else f"{prefix}__{suffix}"
    safe = INVALID_KEY_CHARS.sub("_", raw)
    if len(safe) <= QUESTION_KEY_LIMIT:
        return safe
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=4).hexdigest()
    return f"{safe[: QUESTION_KEY_LIMIT - len(digest) - 1]}_{digest}"


def validate_question(key: str, spec: dict[str, Any]) -> None:
    """Fail locally on schema mistakes instead of paying for an HTTP 400."""
    unknown = set(spec) - ALLOWED_QUESTION_FIELDS
    if unknown:
        raise SystemExit(f"Error: question '{key}' has unsupported field(s): {sorted(unknown)}")
    qtype = spec.get("type")
    if qtype not in ALLOWED_QUESTION_TYPES:
        raise SystemExit(
            f"Error: question '{key}' has type {qtype!r}; "
            f"expected one of {sorted(ALLOWED_QUESTION_TYPES)}"
        )
    if not spec.get("instructions"):
        raise SystemExit(f"Error: question '{key}' is missing 'instructions'")
    if qtype in {"choice", "score"} and not spec.get("criteria"):
        raise SystemExit(f"Error: question '{key}' of type {qtype!r} requires 'criteria'")


def lint_questions(questions: dict[str, Any]) -> list[str]:
    """Report question-design mistakes that predict a useless answer.

    These are the four failure modes seen in practice: a pairwise question the model cannot
    answer in one pass, negated phrasing that pushes the violation into `1 - p`, a missing
    criteria boundary that leaves the threshold undefined, and a self-referential question
    that was never expanded per symbol.
    """
    warnings: list[str] = []
    seen: dict[str, str] = {}
    for key, spec in questions.items():
        if not isinstance(spec, dict):
            continue
        instructions = spec.get("instructions")
        if not isinstance(instructions, str):
            continue
        lowered = instructions.lower()

        for phrase in PAIRWISE_PHRASES:
            if phrase in lowered:
                warnings.append(
                    f"{key}: pairwise phrasing '{phrase}'. The model cannot compare symbols in "
                    "one pass; run --candidates and ask one question per candidate instead."
                )
                break

        if NEGATION_PATTERN.search(instructions) and not BENIGN_NEGATION.search(instructions):
            warnings.append(
                f"{key}: negated phrasing. Ask about the defect positively and invert in code, "
                "otherwise the violation hides in 1 - p."
            )

        if spec.get("type") == "noul" and not spec.get("criteria"):
            warnings.append(
                f"{key}: noul without criteria. Add true_description/false_description so the "
                "boundary between yes and no is defined."
            )

        if SELF_REFERENCE_PATTERN.search(instructions) and not spec.get("per_symbol"):
            warnings.append(
                f"{key}: refers to 'this symbol' but is not per_symbol, so it will be asked once "
                "about the whole shared input."
            )

        previous = seen.get(instructions)
        if previous is not None:
            warnings.append(
                f"{key}: instructions are identical to those of '{previous}'. Two identical "
                "questions produce two identical answers and no new information."
            )
        else:
            seen[instructions] = key
    return warnings


def expand_per_symbol_questions(
    questions: dict[str, Any],
    symbols: list[Symbol],
    force_all: bool = False,
    include_classes: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Clone each `per_symbol` question across every discovered symbol.

    The clone carries the target symbol in `instructions` because the model never
    sees question keys. Without that prefix, every clone would be an identical
    question about the whole file and the localization would be meaningless.

    Returns the expanded question map plus a key -> origin map. The origin map is
    recorded here rather than re-derived by splitting the key on `__`, because a
    base question may itself contain `__` and a shortened key carries a digest
    instead of the full symbol id.
    """
    targets = symbols if include_classes else [s for s in symbols if s.kind in CALLABLE_KINDS]
    expanded: dict[str, dict[str, Any]] = {}
    origins: dict[str, dict[str, Any]] = {}

    for base_key, raw_spec in questions.items():
        if not isinstance(raw_spec, dict):
            raise SystemExit(f"Error: question '{base_key}' must be a JSON object")
        spec: dict[str, Any] = dict(cast("dict[str, Any]", raw_spec))
        per_symbol = bool(spec.pop("per_symbol", False)) or (
            force_all and spec.get("type") == "noul"
        )
        flag_option = spec.pop("flag_option", None)
        canary = bool(spec.pop("canary", False))
        informational = bool(spec.pop("informational", False))
        flag_at = spec.pop("flag_at", None)
        basis = spec.pop("basis", None)
        validate_question(base_key, spec)
        if flag_at is not None:
            if spec.get("type") != "noul":
                raise SystemExit(
                    f"Error: question '{base_key}' sets flag_at but is not a noul question"
                )
            if not isinstance(flag_at, (int, float)) or not 0.01 <= float(flag_at) <= 0.99:
                raise SystemExit(
                    f"Error: question '{base_key}' flag_at must be between 0.01 and 0.99"
                )
            # A lowered threshold widens what counts as a finding, so it must carry its
            # evidence; otherwise tuning becomes indistinguishable from guessing.
            if abs(float(flag_at) - NOUL_FLAG_THRESHOLD) > 1e-9 and not basis:
                raise SystemExit(
                    f"Error: question '{base_key}' lowers flag_at to {flag_at} without a 'basis'. "
                    "State the measurements that justify it."
                )
        if flag_option is not None:
            if spec.get("type") != "choice":
                raise SystemExit(
                    f"Error: question '{base_key}' sets flag_option but is not a choice question"
                )
            if flag_option not in (spec.get("criteria") or {}):
                raise SystemExit(
                    f"Error: question '{base_key}' flag_option {flag_option!r} is not one of its "
                    "criteria options"
                )

        origin_fields: dict[str, Any] = {"question": base_key}
        if flag_option is not None:
            origin_fields["flag_option"] = flag_option
        if flag_at is not None:
            origin_fields["flag_at"] = float(flag_at)
        if canary:
            origin_fields["canary"] = True
        if informational:
            origin_fields["informational"] = True

        if not per_symbol:
            key = make_question_key(base_key)
            expanded[key] = spec
            origins[key] = {**origin_fields, "symbol": None}
            continue

        if not targets:
            print(
                f"warning: '{base_key}' is per_symbol but no AST symbols were found; sending it once",
                file=sys.stderr,
            )
            key = make_question_key(base_key)
            expanded[key] = spec
            origins[key] = {**origin_fields, "symbol": None}
            continue

        instructions = spec["instructions"]
        if not isinstance(instructions, str):
            raise SystemExit(
                f"Error: question '{base_key}' needs plain-string 'instructions' to expand per symbol"
            )
        for symbol in targets:
            clone = dict(spec)
            clone["instructions"] = (
                f"[Target symbol: `{symbol.label}`. Answer only about this symbol.]\n{instructions}"
            )
            key = make_question_key(base_key, symbol.raw_id)
            expanded[key] = clone
            origins[key] = {**origin_fields, "symbol": symbol.raw_id}

    return expanded, origins


def _answer_kind(answer: Any) -> str | None:
    kind = getattr(answer, "type", None)
    if isinstance(kind, str) and kind in ALLOWED_QUESTION_TYPES:
        return kind
    for candidate in ("noul", "choice", "score"):
        if hasattr(answer, candidate):
            return candidate
    return None


def _rounded_map(values: Any) -> dict[str, float]:
    if not values:
        return {}
    return {str(key): round(float(value), 3) for key, value in dict(values).items()}


def _cap(bucket: dict[str, Any]) -> dict[str, Any]:
    """Keep the report proportional to findings so it stays in the caller's context."""
    if len(bucket) <= MAX_REPORT_ITEMS:
        return bucket
    return dict(list(bucket.items())[:MAX_REPORT_ITEMS])


def _merge_answers(
    report: dict[str, Any],
    index: dict[str, Any],
    answers: Any,
    origins: dict[str, dict[str, Any]],
) -> None:
    for key, answer in dict(answers).items():
        kind = _answer_kind(answer)
        origin = origins.get(key, {"question": key, "symbol": None})
        # Record provenance for every answer, not just the flagged ones: calibration has to
        # resolve clean answers back to a question, and those are the negative labels.
        index[key] = origin
        entry: dict[str, Any]

        if kind == "noul":
            probability = float(answer.noul)
            confidence = float(getattr(answer, "confidence", max(probability, 1.0 - probability)))
            entry = {"p_yes": round(probability, 3), "confidence": round(confidence, 3)}
            report["noul_probabilities"][key] = entry["p_yes"]
            if origin.get("informational"):
                # Explanatory answers (e.g. "is this required by a protocol?") are recorded
                # but never counted as findings: yes is not a defect for that question.
                report["informational"][key] = entry["p_yes"]
                continue
            flag_at = float(origin.get("flag_at", NOUL_FLAG_THRESHOLD))
            if probability >= flag_at:
                report["flagged_or_true"][key] = entry
            elif probability <= NOUL_CLEAN_THRESHOLD:
                report["false_or_clean_count"] += 1
                report["clean_keys"].append(key)
            else:
                report["escalate_to_reasoning_llm"][key] = entry
            continue

        if kind == "choice":
            confidence = float(getattr(answer, "confidence", 0.0))
            winner = getattr(answer, "choice", None)
            probabilities = _rounded_map(getattr(answer, "probabilities", None))
            entry = {
                "winner": winner,
                "confidence": round(confidence, 3),
                "probabilities": probabilities,
            }
            report["choices_and_scores"][key] = entry
            # A choice question only flags when the caller named which option means "defect";
            # without that declaration the winner is just a label, not a verdict.
            flag_option = origin.get("flag_option")
            if flag_option and winner == flag_option and confidence >= CHOICE_CONFIDENCE_FLOOR:
                report["flagged_or_true"][key] = {
                    "winner": winner,
                    "confidence": round(confidence, 3),
                    "probability": probabilities.get(str(winner)),
                }
            if confidence < CHOICE_CONFIDENCE_FLOOR:
                report["escalate_to_reasoning_llm"][key] = entry
            continue

        if kind == "score":
            confidence = float(getattr(answer, "confidence", 0.0))
            entry = {
                "expected_score": round(float(getattr(answer, "score", 0.0)), 3),
                "confidence": round(confidence, 3),
                "probabilities": _rounded_map(getattr(answer, "probabilities", None)),
                "legend": {str(k): v for k, v in dict(getattr(answer, "legend", {}) or {}).items()},
            }
            report["choices_and_scores"][key] = entry
            if confidence < SCORE_CONFIDENCE_FLOOR:
                report["escalate_to_reasoning_llm"][key] = entry
            continue

        report["unparsed"][key] = {"repr": repr(answer)[:200]}


def summarize_questions(report: dict[str, Any]) -> dict[str, Any]:
    """Per-question signal, for both yes/no and choice answers.

    A flat band means the question carried no signal: every answer clustered near 0 means
    either the code is clean or the question is blind, and only a known-positive example
    separates those two readings. A wide spread with high values means it is discriminating.
    """
    probabilities = report.get("noul_probabilities", {})
    index = report.get("question_index", {})
    grouped: dict[str, list[float]] = {}
    for key, probability in probabilities.items():
        question = index.get(key, {}).get("question", key)
        grouped.setdefault(question, []).append(probability)

    summary: dict[str, Any] = {}
    for question, values in grouped.items():
        lowest = min(values)
        highest = max(values)
        if highest - lowest >= SIGNAL_SPREAD:
            verdict = "discriminating"
        elif highest <= NOUL_CLEAN_THRESHOLD:
            verdict = "flat-clean"
        elif lowest >= NOUL_FLAG_THRESHOLD:
            verdict = "flat-flagged"
        else:
            verdict = "ambiguous"
        summary[question] = {
            "type": "noul",
            "answers": len(values),
            "min": round(lowest, 3),
            "mean": round(sum(values) / len(values), 3),
            "max": round(highest, 3),
            "spread": round(highest - lowest, 3),
            "verdict": verdict,
        }

    choice_groups: dict[str, list[dict[str, Any]]] = {}
    for key, entry in report.get("choices_and_scores", {}).items():
        question = index.get(key, {}).get("question", key)
        choice_groups.setdefault(question, []).append(entry)
    for question, entries in choice_groups.items():
        if question in summary:
            continue
        winners = [entry.get("winner") for entry in entries]
        flag_option = index.get(
            next(
                (
                    key
                    for key in report.get("choices_and_scores", {})
                    if index.get(key, {}).get("question") == question
                ),
                "",
            ),
            {},
        ).get("flag_option")
        hit = any(
            winner == flag_option and entry.get("confidence", 0) >= CHOICE_CONFIDENCE_FLOOR
            for winner, entry in zip(winners, entries, strict=True)
        )
        if hit:
            verdict = "flagged"
        elif len(set(winners)) > 1:
            verdict = "mixed"
        else:
            verdict = "unanimous"
        summary[question] = {
            "type": "choice",
            "answers": len(entries),
            "winners": {str(winner): winners.count(winner) for winner in set(winners)},
            "mean_confidence": round(
                sum(float(entry.get("confidence", 0.0)) for entry in entries) / len(entries), 3
            ),
            "flag_option": flag_option,
            "verdict": verdict,
        }
    return summary


def estimate_tokens(text: str, questions: dict[str, Any]) -> int:
    """Optimistic input-token estimate: shared input plus every question's instructions."""
    payload = len(text) + sum(
        len(json.dumps(spec, ensure_ascii=False)) for spec in questions.values()
    )
    return math.ceil(payload / CHARS_PER_TOKEN)


def estimate_tokens_conservative(text: str, questions: dict[str, Any]) -> int:
    """Upper-bound estimate. Measured actuals landed between the two ratios."""
    payload = len(text) + sum(
        len(json.dumps(spec, ensure_ascii=False)) for spec in questions.values()
    )
    return math.ceil(payload / CONSERVATIVE_CHARS_PER_TOKEN)


def find_env_file(explicit: str | None) -> Path | None:
    """Locate a `.env` for the API key: explicit path, else CWD and up to 3 parents."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"Error: --env-file {path} does not exist")
        return path
    candidate = Path.cwd().resolve()
    for _ in range(4):
        env_file = candidate / ".env"
        if env_file.is_file():
            return env_file
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return None


def load_env_file(path: Path | None) -> None:
    """Populate missing environment variables from a `.env`; never override real ones."""
    if path is None:
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=" if "=" in line else ":" if ":" in line else ""
        if not separator:
            continue
        name, _, value = line.partition(separator)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def resolve_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    if key.strip().lower() in PLACEHOLDER_KEYS:
        raise SystemExit(
            "Error: GEMINI_API_KEY is missing or still a placeholder. "
            "Set it in the environment or in a .env file (keep that file out of version control)."
        )
    return key


def load_questions(args: argparse.Namespace) -> dict[str, Any]:
    if args.questions_file:
        raw = args.questions_file.read_text(encoding="utf-8")
    elif args.json_questions == "-":
        raw = sys.stdin.read()
    elif args.json_questions:
        raw = args.json_questions
    else:
        raise SystemExit("Error: pass questions with -q <file>, -j '<json>', or -j - for stdin")
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: questions are not valid JSON: {exc}") from exc
    if not isinstance(questions, dict):
        raise SystemExit("Error: questions must be a JSON object mapping key -> question")
    return cast("dict[str, Any]", questions)


def build_report(
    shared_input: str,
    questions: dict[str, Any],
    origins: dict[str, dict[str, Any]],
    model: str,
    include_clean_keys: bool,
) -> dict[str, Any]:
    """Send questions in batches of 1,000 and triage every answer."""
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "Error: the 'google-genai' package is missing. Run this script with "
            "`uv run verify_claims.py ...` so its inline dependencies are installed."
        ) from exc

    client = genai.Client(http_options={"api_version": "v1beta"})
    items = list(questions.items())
    batches = math.ceil(len(items) / MAX_QUESTIONS_PER_REQUEST)

    if batches > 1:
        print(
            f"note: {len(items)} questions split into {batches} requests; "
            "the shared input is billed once per request.",
            file=sys.stderr,
        )

    index: dict[str, Any] = {}
    report: dict[str, Any] = {
        "flagged_or_true": {},
        "false_or_clean_count": 0,
        "escalate_to_reasoning_llm": {},
        "choices_and_scores": {},
        "unparsed": {},
        "clean_keys": [],
        "noul_probabilities": {},
        "informational": {},
        "usage": {"requests": 0, "questions": len(items), "total_input_tokens": 0},
        "meta": {"model": model, "batches": batches},
    }

    for start in range(0, len(items), MAX_QUESTIONS_PER_REQUEST):
        batch = dict(items[start : start + MAX_QUESTIONS_PER_REQUEST])
        try:
            result = client.decisions.create(
                model=model,
                input=shared_input,
                questions=batch,
            )
        except Exception as exc:  # surface the API's own message instead of a bare traceback
            raise SystemExit(
                f"Error: Decisions API request failed ({type(exc).__name__}): {exc}\n"
                "Hint: HTTP 400 usually means a schema/limit violation (1-1,000 questions, "
                f"keys matching [A-Za-z0-9_-]+ up to 64 chars, {MAX_CONTEXT_TOKENS} token context)."
            ) from exc

        _merge_answers(report, index, getattr(result, "answers", {}), origins)
        usage = getattr(result, "usage", None)
        report["usage"]["requests"] += 1
        if usage is not None:
            report["usage"]["total_input_tokens"] += int(
                getattr(usage, "total_input_tokens", 0) or 0
            )

    report["flagged_or_true"] = _cap(report["flagged_or_true"])
    report["escalate_to_reasoning_llm"] = _cap(report["escalate_to_reasoning_llm"])
    report["choices_and_scores"] = _cap(report["choices_and_scores"])
    report["question_index"] = index
    report["question_summary"] = summarize_questions(report)
    for question, stats in report["question_summary"].items():
        if stats["verdict"] == "flat-clean":
            print(
                f"warning: '{question}' produced no signal (max p={stats['max']}, spread="
                f"{stats['spread']}). The code may be clean, or the question may be blind; only a "
                "known-positive example separates those readings.",
                file=sys.stderr,
            )
    if not include_clean_keys:
        report.pop("clean_keys", None)
    return report


def dry_run_payload(
    file_paths: list[Path],
    shared_input: str,
    symbols: list[Symbol],
    questions: dict[str, Any],
    preview: int,
) -> dict[str, Any]:
    tokens = estimate_tokens(shared_input, questions)
    conservative = estimate_tokens_conservative(shared_input, questions)
    return {
        "mode": "dry_run",
        "files": [str(path) for path in file_paths],
        "symbols_found": len(symbols),
        "symbol_kinds": {
            kind: sum(1 for s in symbols if s.kind == kind)
            for kind in sorted({s.kind for s in symbols})
        },
        "questions_total": len(questions),
        "requests_required": math.ceil(len(questions) / MAX_QUESTIONS_PER_REQUEST) or 1,
        "estimated_input_tokens": tokens,
        "conservative_input_tokens": conservative,
        "context_limit_tokens": MAX_CONTEXT_TOKENS,
        "within_context_limit": conservative <= MAX_CONTEXT_TOKENS,
        "duplicate_keys": len(questions) != len(set(questions)),
        "sample_questions": dict(list(questions.items())[: max(preview, 0)]),
    }


def _label_pairs(
    labels: dict[str, Any],
    probabilities: dict[str, float],
    index: dict[str, Any],
) -> dict[str, list[tuple[bool, float]]]:
    """Group labeled answers into (expected, probability) pairs per base question.

    Labels are `{base_question: {symbol_pattern: expected}}`; the pattern `*` matches every
    symbol of that question, and a bare boolean applies to the unexpanded question.
    """
    grouped: dict[str, list[tuple[bool, float]]] = {}
    for base_key, spec in labels.items():
        entries = spec if isinstance(spec, dict) else {"*": spec}
        for pattern, expected in entries.items():
            if not isinstance(expected, bool):
                continue
            for key, probability in probabilities.items():
                origin = index.get(key, {})
                if origin.get("question") != base_key:
                    continue
                if pattern != "*" and pattern not in (origin.get("symbol") or ""):
                    continue
                grouped.setdefault(base_key, []).append((expected, probability))
    return grouped


def _sweep(pairs: list[tuple[bool, float]]) -> list[dict[str, Any]]:
    """Precision and recall at each threshold, counting `probability >= threshold` as yes."""
    rows: list[dict[str, Any]] = []
    for step in range(1, 20):
        threshold = step / 20
        tp = fp = fn = tn = 0
        for expected, probability in pairs:
            predicted = probability >= threshold
            if predicted and expected:
                tp += 1
            elif predicted:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "threshold": round(threshold, 2),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": None if precision is None else round(precision, 3),
                "recall": None if recall is None else round(recall, 3),
                "f1": round(f1, 3),
            }
        )
    return rows


def calibrate(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Score a finished run against labeled ground truth and pick one threshold per question.

    Both classes must be present: with only positives, every threshold has recall 1.0 and
    the sweep says nothing about false positives.
    """
    probabilities = report.get("noul_probabilities", {})
    index = report.get("question_index", {})
    grouped = _label_pairs(labels, probabilities, index)
    out: dict[str, Any] = {"per_question": {}, "warnings": []}

    for base_key, pairs in grouped.items():
        positives = [probability for expected, probability in pairs if expected]
        negatives = [probability for expected, probability in pairs if not expected]
        rows = _sweep(pairs)
        best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
        out["per_question"][base_key] = {
            "labeled": len(pairs),
            "positives": len(positives),
            "negatives": len(negatives),
            "mean_p_positives": (
                round(sum(positives) / len(positives), 3) if positives else None
            ),
            "mean_p_negatives": (
                round(sum(negatives) / len(negatives), 3) if negatives else None
            ),
            "best_threshold": best["threshold"],
            "best_f1": best["f1"],
            "sweep": rows,
        }
        if not positives or not negatives:
            out["warnings"].append(
                f"{base_key}: needs both positive and negative labels to choose a threshold"
            )

    if not grouped:
        out["warnings"].append(
            "no labeled answers matched; label keys must equal question keys and every symbol "
            "pattern must appear in question_index"
        )
    return out


def canary_check(report: dict[str, Any], canary_ids: set[str]) -> dict[str, Any]:
    """Did each question fire on a symbol that is known to violate it?

    This is the only way to separate "the code is clean" from "the question cannot fire".
    Questions with no canary answers are untested, not passing.
    """
    per_question: dict[str, list[float]] = {}
    for key, probability in report.get("noul_probabilities", {}).items():
        origin = report.get("question_index", {}).get(key, {})
        if origin.get("canary") or origin.get("symbol") in canary_ids:
            per_question.setdefault(origin.get("question", key), []).append(probability)
    for key, entry in report.get("choices_and_scores", {}).items():
        origin = report.get("question_index", {}).get(key, {})
        if not (origin.get("canary") or origin.get("symbol") in canary_ids):
            continue
        signal = 1.0 if entry.get("winner") == origin.get("flag_option") else 0.0
        per_question.setdefault(origin.get("question", key), []).append(signal)
    return {
        question: {
            "canary_answers": len(values),
            "max_signal": round(max(values), 3),
            "fired": max(values) >= NOUL_FLAG_THRESHOLD,
        }
        for question, values in per_question.items()
    }


def build_digest(report: dict[str, Any], limit: int = 40) -> str:
    """Compact markdown for the calling agent: findings and escalations only.

    The full report keeps a probability for every answer and a provenance row for every key,
    which is what calibration needs and what a reader does not: measured on a 146-question
    run, 75% of the report was those two maps and the findings were 2%. A caller that reads
    the raw JSON pays for all of it, so this strips the bookkeeping and keeps the decisions.
    """
    index = report.get("question_index", {})
    usage = report.get("usage", {})
    flagged = report.get("flagged_or_true", {})
    escalated = report.get("escalate_to_reasoning_llm", {})
    lines = ["# Decision audit digest", ""]
    lines.append(
        f"- questions {usage.get('questions', 0)} · requests {usage.get('requests', 0)} · "
        f"input tokens {usage.get('total_input_tokens', 0)}"
    )
    lines.append(
        f"- flagged {len(flagged)} · escalated {len(escalated)} · "
        f"clean {report.get('false_or_clean_count', 0)} · "
        f"informational {len(report.get('informational', {}))}"
    )

    def signal(entry: dict[str, Any]) -> str:
        if "p_yes" in entry:
            return f"p={entry['p_yes']:.2f}"
        return f"{entry.get('winner')} conf={entry.get('confidence', 0):.2f}"

    lines.append("\n## Flagged: act on these")
    if not flagged:
        lines.append("- none")
    for key, entry in list(flagged.items())[:limit]:
        origin = index.get(key, {})
        symbol = origin.get("symbol") or "(whole input)"
        lines.append(f"- {signal(entry)} · {origin.get('question', key)} · `{symbol}`")
    if len(flagged) > limit:
        lines.append(f"- … {len(flagged) - limit} more in the report file")

    lines.append("\n## Escalated: the model could not decide, read these")
    if not escalated:
        lines.append("- none")
    grouped: dict[str, list[tuple[float, str]]] = {}
    for key, entry in escalated.items():
        origin = index.get(key, {})
        symbol = origin.get("symbol") or "(whole input)"
        raw = entry.get("p_yes", entry.get("probability") or entry.get("confidence") or 0.0)
        grouped.setdefault(origin.get("question", key), []).append((float(raw), symbol))
    for question, items in grouped.items():
        lines.append(f"\n### {question} ({len(items)})")
        for value, symbol in sorted(items, key=lambda item: -item[0])[:limit]:
            lines.append(f"- {value:.2f} · `{symbol}`")
        if len(items) > limit:
            lines.append(f"- … {len(items) - limit} more in the report file")

    summary = report.get("question_summary", {})
    if summary:
        lines.append("\n## Question signal")
        for question, stats in summary.items():
            if stats.get("type") == "choice":
                lines.append(
                    f"- {question}: {stats.get('verdict')} "
                    f"(mean conf {stats.get('mean_confidence')})"
                )
            else:
                lines.append(
                    f"- {question}: {stats.get('verdict')} "
                    f"(p {stats.get('min')} to {stats.get('max')}, mean {stats.get('mean')})"
                )

    canary = report.get("canary", {})
    if canary:
        lines.append("\n## Canary")
        for question, info in canary.items():
            mark = "fired" if info.get("fired") else "did NOT fire"
            lines.append(f"- {question}: {mark} (signal {info.get('max_signal')})")

    return "\n".join(lines)


def fail_on_missing_files(file_paths: list[Path]) -> None:
    """Abort on missing inputs, suggesting near-matches instead of echoing a bare path."""
    missing = [path for path in file_paths if not path.is_file()]
    if not missing:
        return
    lines = [f"Error: file(s) not found: {[str(path) for path in missing]}"]
    for path in missing:
        parent = path.parent
        if not parent.is_dir():
            continue
        names = sorted(entry.name for entry in parent.iterdir() if entry.is_file())
        close = difflib.get_close_matches(path.name, names, n=3, cutoff=0.5)
        if close:
            lines.append(f"  {path.name}: did you mean {[str(parent / name) for name in close]}?")
        elif names:
            preview = names[:5]
            more = " ..." if len(names) > len(preview) else ""
            lines.append(f"  available in {parent}/: {preview}{more}")
    raise SystemExit("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate many atomic questions over shared files with gemini-decision-1.",
        epilog=(
            "Always run with --dry-run first: it prints the AST index summary, the expanded "
            "question count, the required request count, and a token estimate without spending "
            "anything."
        ),
    )
    parser.add_argument("files", nargs="+", type=Path, help="Text, source, .tex, or .md files.")
    parser.add_argument("-q", "--questions-file", type=Path, help="JSON file of questions.")
    parser.add_argument("-j", "--json-questions", help="Inline JSON questions, or '-' for stdin.")
    parser.add_argument(
        "--all-noul-per-symbol",
        action="store_true",
        help=(
            "Clone EVERY noul question across every discovered symbol, even those without "
            "'per_symbol': true. Multiplies the question count by the symbol count; confirm "
            "the total with --dry-run first."
        ),
    )
    parser.add_argument(
        "--include-classes",
        action="store_true",
        help="Also target classes (not just functions and methods) when expanding per symbol.",
    )
    parser.add_argument("--no-ast", action="store_true", help="Skip AST index injection.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the AST summary, expanded questions, and token estimate; no API call.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="How many expanded questions to print in --dry-run (default: 5).",
    )
    parser.add_argument(
        "--list-symbols",
        action="store_true",
        help="Print every discovered symbol id and exit; no API call, no questions needed.",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Print duplication candidates (identical identifier-erased AST shapes); no API call.",
    )
    parser.add_argument(
        "--emit-questions",
        type=Path,
        metavar="PATH",
        help="Write a questions file for the duplication candidates (implies --candidates).",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.60,
        metavar="RATIO",
        help="Jaccard threshold for near-duplicate candidates (default: 0.60).",
    )
    parser.add_argument(
        "--calibrate",
        type=Path,
        metavar="LABELS_JSON",
        help="Score this run against labeled ground truth and pick a threshold per question.",
    )
    parser.add_argument(
        "--strict-questions",
        action="store_true",
        help="Treat question-lint warnings as errors, before any API call is made.",
    )
    parser.add_argument(
        "--canary",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "File of known violations, added to the shared input and to per-symbol targets. "
            "A question that never fires on it is proven blind. Repeatable."
        ),
    )
    parser.add_argument(
        "--env-file", help="Path to a .env holding GEMINI_API_KEY."
    )
    parser.add_argument("--model", default=MODEL, help=f"Model id (default: {MODEL}).")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this file.")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Include the keys that passed cleanly (hidden by default to save context).",
    )
    parser.add_argument(
        "--stdout",
        choices=("digest", "json"),
        default=None,
        help=(
            "What to print to stdout. Default: digest when --output is set, json otherwise. "
            "'digest' needs --output so the full report is never lost."
        ),
    )
    parser.add_argument(
        "--digest-limit",
        type=int,
        default=40,
        metavar="N",
        help="Maximum rows per group in the digest (default: 40).",
    )
    parser.add_argument(
        "--fail-on-flagged",
        action="store_true",
        help="Exit 1 when at least one question was flagged (useful as a CI gate).",
    )
    args = parser.parse_args()

    canaries: list[Path] = list(args.canary or [])
    fail_on_missing_files(args.files + canaries)
    payload_paths = list(args.files)
    for canary in canaries:
        if canary not in payload_paths:
            payload_paths.append(canary)
    canary_stems = {path.stem for path in canaries}

    if args.stdout == "digest" and args.output is None:
        raise SystemExit(
            "Error: --stdout digest needs --output, otherwise the full report is lost. "
            "Pass --output PATH."
        )
    stdout_format = args.stdout or ("digest" if args.output else "json")

    if args.list_symbols:
        for symbol in collect_symbols(payload_paths):
            print(f"{symbol.kind}\t{symbol.raw_id}")
        return

    if args.candidates or args.emit_questions:
        clusters = candidate_clusters(
            collect_symbols(payload_paths),
            args.include_classes,
            args.similarity,
            canary_stems,
        )
        if args.emit_questions:
            args.emit_questions.write_text(
                json.dumps(cluster_questions(clusters["clusters"]), indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.emit_questions}", file=sys.stderr)
        if args.candidates:
            print(json.dumps(clusters, indent=2, ensure_ascii=False))
        return

    base_questions = load_questions(args)
    lint_warnings = lint_questions(base_questions)
    for warning in lint_warnings:
        print(f"question-lint: {warning}", file=sys.stderr)
    if lint_warnings and args.strict_questions:
        raise SystemExit(
            "Error: question lint failed under --strict-questions; fix the warnings above."
        )
    shared_input, symbols = build_shared_context(
        payload_paths, include_ast=not args.no_ast
    )
    if canary_stems:
        symbols = [
            replace(symbol, canary=True) if symbol.module in canary_stems else symbol
            for symbol in symbols
        ]
    questions, origins = expand_per_symbol_questions(
        base_questions,
        symbols,
        force_all=args.all_noul_per_symbol,
        include_classes=args.include_classes,
    )

    if not questions:
        raise SystemExit("Error: no questions to evaluate after expansion.")
    if len(questions) != len(set(questions)):
        raise SystemExit("Error: internal key collision while expanding questions; please report.")

    if args.dry_run:
        payload = dry_run_payload(args.files, shared_input, symbols, questions, args.preview)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if not payload["within_context_limit"]:
            print(
                f"warning: estimated {payload['estimated_input_tokens']} input tokens exceeds the "
                f"{MAX_CONTEXT_TOKENS} limit; split the file list across runs.",
                file=sys.stderr,
            )
        return

    if estimate_tokens_conservative(shared_input, questions) > MAX_CONTEXT_TOKENS:
        raise SystemExit(
            f"Error: estimated input exceeds the {MAX_CONTEXT_TOKENS} token context window. "
            "Pass fewer files per run (the shared input is not split automatically)."
        )

    load_env_file(find_env_file(args.env_file))
    resolve_api_key()

    report = build_report(
        shared_input,
        questions,
        origins,
        model=args.model,
        include_clean_keys=args.explain,
    )
    if canary_stems:
        report["canary"] = canary_check(
            report, {symbol.raw_id for symbol in symbols if symbol.canary}
        )
        for question in report.get("question_summary", {}):
            info = report["canary"].get(question)
            if info is None:
                print(
                    f"warning: canary: '{question}' is untested; it produced no canary answers, "
                    "so a clean result cannot be told apart from a blind question.",
                    file=sys.stderr,
                )
            elif info["fired"]:
                print(
                    f"canary: '{question}' fired (signal {info['max_signal']})", file=sys.stderr
                )
            else:
                print(
                    f"warning: canary: '{question}' did not fire on its canary "
                    f"(max signal {info['max_signal']}); the question is proven blind.",
                    file=sys.stderr,
                )
    if args.calibrate:
        labeled = json.loads(args.calibrate.read_text(encoding="utf-8"))
        if not isinstance(labeled, dict):
            raise SystemExit("Error: --calibrate expects a JSON object of labels")
        report["calibration"] = calibrate(report, labeled)
        summary = {
            name: {
                "best_threshold": value["best_threshold"],
                "best_f1": value["best_f1"],
                "labeled": value["labeled"],
            }
            for name, value in report["calibration"]["per_question"].items()
        }
        print(f"calibration: {json.dumps(summary)}", file=sys.stderr)
        for warning in report["calibration"]["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(
            f"wrote {args.output} | flagged={len(report['flagged_or_true'])} "
            f"escalated={len(report['escalate_to_reasoning_llm'])} "
            f"clean={report['false_or_clean_count']} "
            f"input_tokens={report['usage']['total_input_tokens']}",
            file=sys.stderr,
        )
    if stdout_format == "digest":
        print(build_digest(report, args.digest_limit))
    else:
        print(serialized)

    if args.fail_on_flagged and report["flagged_or_true"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
