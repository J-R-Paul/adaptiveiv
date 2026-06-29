from __future__ import annotations

from dataclasses import dataclass
import re

import patsy


@dataclass(frozen=True)
class FormulaSpec:
    dependent: str
    exog: list[str]
    endogenous: str
    instrument: str
    add_intercept: bool


_IV_BLOCK = re.compile(r"\[(?P<endog>[^\[\]~]+)~(?P<instr>[^\[\]~]+)\]")


def _clean_term(term: str) -> str:
    term = term.strip()
    quoted = re.fullmatch(r'Q\(["\'](?P<name>.+)["\']\)', term)
    if quoted:
        return quoted.group("name")
    return term


def _split_terms(rhs: str) -> list[str]:
    terms = []
    for raw in rhs.split("+"):
        term = _clean_term(raw)
        if term:
            terms.append(term)
    return terms


def _terms_from_patsy(desc: patsy.ModelDesc) -> tuple[list[str], bool]:
    exog: list[str] = []
    add_intercept = False
    for term in desc.rhs_termlist:
        if len(term.factors) == 0:
            add_intercept = True
            continue
        if len(term.factors) != 1:
            raise ValueError("Interaction terms are not supported in adaptiveiv formulas.")
        factor = term.factors[0]
        name = factor.code
        exog.append(_clean_term(name))
    return exog, add_intercept


def parse_iv_formula(formula: str) -> FormulaSpec:
    matches = list(_IV_BLOCK.finditer(formula))
    if len(matches) != 1:
        raise ValueError("Formula must contain exactly one IV block like [W ~ Z].")

    left, rhs = formula.split("~", 1)
    dependent = _clean_term(left)
    if not dependent:
        raise ValueError("Formula must include a dependent variable.")

    rhs_match = _IV_BLOCK.search(rhs)
    if rhs_match is None:
        raise ValueError("Formula must contain exactly one IV block like [W ~ Z].")

    match = matches[0]
    endogenous_terms = _split_terms(match.group("endog"))
    instrument_terms = _split_terms(match.group("instr"))
    if len(endogenous_terms) != 1:
        raise ValueError("adaptiveiv currently supports one endogenous variable.")
    if len(instrument_terms) != 1:
        raise ValueError("adaptiveiv currently supports one excluded instrument.")

    rhs_without_iv = rhs[: rhs_match.start()] + rhs[rhs_match.end() :]
    rhs_for_patsy = re.sub(r"(^\s*\+|\+\s*$)", "", rhs_without_iv).strip() or "1"
    desc = patsy.ModelDesc.from_formula(f"{left.strip()} ~ {rhs_for_patsy}")
    exog, add_intercept = _terms_from_patsy(desc)

    return FormulaSpec(
        dependent=dependent,
        exog=exog,
        endogenous=endogenous_terms[0],
        instrument=instrument_terms[0],
        add_intercept=add_intercept,
    )
