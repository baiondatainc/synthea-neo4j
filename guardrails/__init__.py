from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class GuardrailResult(Generic[T]):
    ok: bool
    payload: T
    reason: str = ""


from guardrails.input import check_input
from guardrails.cypher import check_cypher
from guardrails.output import redact_text, redact_rows

__all__ = [
    "GuardrailResult",
    "check_input",
    "check_cypher",
    "redact_text",
    "redact_rows",
]
