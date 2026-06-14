"""
Behaviour cohorts.

V0: instead of running k-means at build time, derive a coarse behaviour-cohort
label deterministically from the flags already on each Patient node. This
keeps the v2 retriever path useful without requiring a separate ML batch job.

If you later want richer cohorts (k-means on financial features, community
detection on the call graph, etc.), replace `assign_cohort` with a real
batch job and write the label back to `Patient.behavior_cohort`.
"""
from __future__ import annotations


COHORTS = {
    "self_pay_high_balance": "Self-pay patient with high outstanding balance.",
    "catastrophe":           "Catastrophe case — large unrecoverable balance.",
    "friction":              "Friction case — repeated touches, slow to pay.",
    "fully_covered":         "Fully covered by insurance, low patient balance.",
    "clean_account":         "Clean account — paid on time, no friction.",
    "bad_debt":              "Account written off to bad debt or collections.",
    "uncontacted":           "Never contacted via call campaigns.",
    "default":               "General patient with mixed profile.",
}


def assign_cohort(row: dict) -> str:
    """Return a single behaviour cohort label for a Patient property row.
    Earlier rules win; the precedence reflects business priority — bad-debt
    and catastrophe trump everything else."""
    bad_debt = float(row.get("adj_bad_debt") or 0) > 0
    if bad_debt:
        return "bad_debt"
    if row.get("is_catastrophe"):
        return "catastrophe"
    if row.get("is_friction"):
        return "friction"
    if row.get("is_self_pay") and float(row.get("outstanding_balance") or 0) > 1000:
        return "self_pay_high_balance"
    if row.get("is_fully_covered"):
        return "fully_covered"
    if row.get("is_clean"):
        return "clean_account"
    if not row.get("has_any_calls"):
        return "uncontacted"
    return "default"
