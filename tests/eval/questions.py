"""Parses docs/questions-eval.md into typed Question records."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Resolve docs/questions-eval.md from repo root. Override with EVAL_QUESTIONS_PATH.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = _REPO_ROOT / "docs" / "questions-eval.md"
print(f"-------------------- Loading eval questions from {DEFAULT_PATH} (override with EVAL_QUESTIONS_PATH)")

CATEGORY_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
# Question line: "1. <text>. **[E]**" possibly with trailing notes after the tag.
QUESTION_RE = re.compile(r"^(\d+)\.\s+(.+?)\s+\*\*\[(E|M|H)\]\*\*")


@dataclass(frozen=True)
class Question:
    number: int
    text: str
    difficulty: str  # "E" | "M" | "H"
    category_number: int
    category: str

    @property
    def slug(self) -> str:
        cat = re.sub(r"[^a-z0-9]+", "-", self.category.lower()).strip("-")
        return f"q{self.number:03d}_{self.difficulty}_{cat}"


def load_questions(path: Path | str | None = None) -> list[Question]:
    p = Path(path or os.getenv("EVAL_QUESTIONS_PATH") or DEFAULT_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Eval questions file not found: {p}")

    questions: list[Question] = []
    cat_num: int | None = None
    cat_name: str = ""

    for line in p.read_text().splitlines():
        m_cat = CATEGORY_RE.match(line)
        if m_cat:
            cat_num = int(m_cat.group(1))
            cat_name = m_cat.group(2).strip()
            continue
        m_q = QUESTION_RE.match(line)
        if m_q:
            if cat_num is None:
                # Question appeared before any category header — shouldn't happen.
                continue
            questions.append(
                Question(
                    number=int(m_q.group(1)),
                    text=m_q.group(2).strip(),
                    difficulty=m_q.group(3),
                    category_number=cat_num,
                    category=cat_name,
                )
            )

    return questions


if __name__ == "__main__":
    qs = load_questions()
    print(f"Loaded {len(qs)} questions")
    for q in qs[:3] + qs[-2:]:
        print(f"  [{q.difficulty}] {q.number:>3}. ({q.category}) {q.text[:60]}")