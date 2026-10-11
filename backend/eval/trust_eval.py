"""
trust_eval.py — does the trust detector actually detect?

The pipeline ships five gates, ten gold-free checks and a verdict, none of which has a
number attached. The selective-prediction literature measures exactly this with
AUROC — the probability that a correct answer is ranked above an incorrect one
— and reports ~0.61–0.68 for self-consistency signals and ~0.72–0.78 for
LLM-judge verification. This module puts our own gates on that scale.

It replays the labelled BIRD records already in ``outputs/`` through the SAME
grading function the live pipeline uses, so what is measured is the shipped
detector rather than a re-implementation of it.

    python -m backend.eval.trust_eval                    # print the report
    python -m backend.eval.trust_eval --output outputs/trust_eval.json

No API key, no database, no new dependency.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
Two kinds of run land in ``outputs/``:

  * **`bird --trust-eval`** stored the real verification block at
    generation time — every check that needs the executed rows, plus the
    back-translation. That block is used as-is.
  * **Everything older** is reconstructed from what it kept: `exec_vote`,
    `valid`, `pred_sql`, `pred_row_count`, `query_plan`. Enough for four of the
    five gates and five of the ten checks; the rows are gone and cannot be
    recovered from disk.

The report follows the DATA, not a constant: a gate skipped on every record is
listed under `not_measured` rather than scored, because a gate that was never
exercised must not be presented as one that was found worthless. The three
value-grounded checks stay unmeasured either way — ValueIndex speaks to
PostgreSQL and the benchmark runs on SQLite.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from backend.agents import risk as R
from backend.agents import trust as T

_OUTPUTS = Path(__file__).resolve().parent.parent.parent / "outputs"

# Reported per RISK CODE rather than per check function: the code is what the
# user is shown and what a finding is argued about, and one check can emit
# several. test_trust_eval.py asserts these two lists together cover every code
# risk.py can produce, so a new check cannot go silently unreported.
#
# Emitted by checks needing only the SQL, question, row count and plan.
REPLAYABLE_CODES = (
    "zero_rows",                        # check_zero_rows
    "missing_group_by",                 # check_aggregate_grain
    "aggregate_mismatch",               # check_aggregate_intent
    "extremum_without_grouping",        # ″
    "percentage_without_division",      # ″
    "limit_one_on_plural",              # check_limit_one_on_plural
    "plan_aggregate_mismatch",          # check_plan_conformance
    "plan_grain_mismatch",              # ″
    "plan_group_by_dropped",            # ″
)
# Emitted by checks needing the executed rows, retrieved tables or value index.
UNMEASURED_CODES = ("ungrounded_literal", "dropped_constraint", "unknown_value",
                    "cross_join", "arity_mismatch", "out_of_schema")
UNMEASURED_GATES = ("back_translation",)

# A gate's status as a score, so a three-level signal can be ranked. `skipped`
# scores as a pass: the gate found nothing wrong because it did not look, and
# penalising that would measure availability rather than discrimination.
_STATUS_SCORE = {T.OK: 1.0, T.SKIPPED: 1.0, T.WARN: 0.5, T.FAIL: 0.0}
_LEVEL_SCORE = {T.PASSED: 1.0, T.REVIEW: 0.5, T.UNTRUSTED: 0.0,
                T.NOT_GRADED: 1.0}


# ── metrics ───────────────────────────────────────────────────────────────────

def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """
    Probability that a randomly chosen correct answer outranks an incorrect one.

    Mann-Whitney U over average ranks, so the coarse 3-level scores these gates
    emit are handled correctly — the naive formulation would treat every tie as
    a win and inflate the figure. Returns None when one class is absent, which
    is a fact about the data rather than a score of 0.5.
    """
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n = len(pairs)
    if n == 0:
        return None
    pos = sum(1 for _s, y in pairs if y)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank of the tie run
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    rank_sum_pos = sum(r for r, (_s, y) in zip(ranks, pairs) if y)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def flag_rates(fired: Sequence[bool], correct: Sequence[int]) -> Dict[str, Any]:
    """
    How good is one check as an accusation?

    precision   — of the answers it flagged, how many were genuinely wrong
    recall      — of the wrong answers, how many it caught
    false_alarm — of the CORRECT answers, how many it flagged anyway

    False alarm is reported separately from precision because it is the number
    that decides whether a user keeps reading the warnings.
    """
    tp = sum(1 for f, c in zip(fired, correct) if f and not c)
    fp = sum(1 for f, c in zip(fired, correct) if f and c)
    wrong = sum(1 for c in correct if not c)
    right = sum(1 for c in correct if c)
    n_fired = tp + fp
    return {
        "fired": n_fired,
        "precision_pct": round(100 * tp / n_fired, 1) if n_fired else None,
        "recall_pct": round(100 * tp / wrong, 1) if wrong else None,
        "false_alarm_pct": round(100 * fp / right, 1) if right else None,
    }


def risk_coverage(scores: Sequence[float], labels: Sequence[int]
                  ) -> List[Dict[str, Any]]:
    """
    Accuracy against the proportion of questions answered, as the abstention
    threshold moves — the standard selective-prediction curve.

    A useful detector buys accuracy by declining: if selective accuracy is flat
    as coverage drops, the ranking carries no information and abstention is
    only losing answers.
    """
    ranked = sorted(zip(scores, labels), key=lambda p: -p[0])
    out: List[Dict[str, Any]] = []
    correct = 0
    for i, (score, y) in enumerate(ranked, start=1):
        correct += y
        # One point per threshold, i.e. at each distinct score boundary.
        if i == len(ranked) or score != ranked[i][0]:
            out.append({
                "threshold": round(score, 3),
                "coverage_pct": round(100 * i / len(ranked), 1),
                "selective_accuracy_pct": round(100 * correct / i, 1),
                "answered": i,
            })
    return out


# ── replay ────────────────────────────────────────────────────────────────────

def _replayable_risks(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The gold-free checks a stored record contains enough state to re-run."""
    sql = rec.get("pred_sql") or ""
    question = rec.get("question") or ""
    n = rec.get("pred_row_count")
    # Only emptiness matters to check_zero_rows; a placeholder stands in for
    # row content the records do not keep.
    rows = [] if n == 0 else ([{"c": 1}] * min(n, 3) if n else None)
    out: List[Dict[str, Any]] = []
    for run in (lambda: R.check_zero_rows(rows),
                lambda: R.check_aggregate_grain(sql, question),
                lambda: R.check_aggregate_intent(sql, question),
                lambda: R.check_limit_one_on_plural(sql, question),
                lambda: R.check_plan_conformance(sql, rec.get("query_plan"))):
        try:
            out.extend(run() or [])
        except Exception:
            pass
    return out


def verification_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    The verification block for one record.

    A run made with ``bird --trust-eval`` stored the real one at
    generation time — including the checks that need the executed rows, and
    the back-translation, none of which can be recovered from disk afterwards.
    Prefer it. Older runs are reconstructed from what they did keep, which is
    strictly less.
    """
    stored = rec.get("verification")
    if isinstance(stored, dict) and stored:
        return stored

    vote = rec.get("exec_vote") or {}
    executed = int(vote.get("executed") or 0)
    valid = bool(rec.get("valid"))
    return {
        "candidates": int(vote.get("candidates") or 0),
        "executed": executed,
        "agreement": int(vote.get("winner_votes") or 0),
        # `groups` is a COUNT in the stored records, not the group list the
        # live block carries — reading it as a list silently crashed.
        "distinct_result_sets": int(vote.get("groups") or 0),
        "decided_by": "plurality" if executed else "none_executed",
        "validated": valid,
        # The records store one boolean, not the three sub-checks; report it
        # as the semantic check, which is the one it stands for.
        "validation_checks": {"semantic_ok": valid},
        "risks": _replayable_risks(rec) if rec.get("pred_sql") else [],
        "refine_rounds": 1 if rec.get("refined") else 0,
        "back_translation": None,          # not stored — gate is not measured
    }


def load_records(pattern: str = "bird_*.json") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(_OUTPUTS / pattern))):
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except Exception as exc:
            print(f"  [skip] {Path(path).name}: {exc}")
            continue
        for rec in doc.get("questions") or []:
            rec = dict(rec)
            rec["_run"] = Path(path).name
            rec["_model"] = (doc.get("run_config") or {}).get("model", "?")
            out.append(rec)
    return out


def evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """AUROC per gate, per-check rates, and the risk–coverage curve."""
    labels: List[int] = []
    verdict_scores: List[float] = []
    gate_scores: Dict[str, List[float]] = defaultdict(list)
    check_fired: Dict[str, List[bool]] = defaultdict(list)
    seen_codes: set = set()
    gate_evidence: Dict[str, bool] = defaultdict(bool)
    verdict_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    for rec in records:
        ev = verification_from_record(rec)
        graded = T.grade(ev)
        correct = 1 if (rec.get("ex") or 0) >= 1 else 0

        labels.append(correct)
        verdict_scores.append(_LEVEL_SCORE[graded["level"]])
        verdict_counts[graded["level"]][0] += 1
        verdict_counts[graded["level"]][1] += correct

        for gate in graded["gates"]:
            gate_scores[gate["id"]].append(_STATUS_SCORE[gate["status"]])
            # A gate SKIPPED on every record was never exercised; its 0.5
            # AUROC would otherwise read as "measured and worthless" rather
            # than "never ran".
            if gate["status"] != T.SKIPPED:
                gate_evidence[gate["id"]] = True
        codes = {str(r.get("code", "")) for r in ev["risks"]}
        seen_codes.update(codes)
        for code in REPLAYABLE_CODES + UNMEASURED_CODES:
            check_fired[code].append(code in codes)

    total = len(labels)
    # What is measurable depends on the runs loaded, not on a constant: a
    # --trust-eval run carries the judge gate and the rows-based checks, an
    # older one does not. A fixed list would understate a fresh run and
    # overstate an old one.
    measured_gates = {gid for gid, seen in gate_evidence.items() if seen}
    unmeasured_codes = [c for c in UNMEASURED_CODES if c not in seen_codes]
    return {
        "total": total,
        "correct": sum(labels),
        "runs": sorted({r.get("_run", "?") for r in records}),
        "models": sorted({r.get("_model", "?") for r in records}),
        "verdict": {
            "auroc": auroc(verdict_scores, labels),
            "bands": {
                lvl: {"n": n, "correct": c,
                      "share_pct": round(100 * n / total, 1) if total else None,
                      "accuracy_pct": round(100 * c / n, 1) if n else None}
                for lvl, (n, c) in sorted(verdict_counts.items())
            },
        },
        "gates": {
            gid: {"auroc": auroc(scores, labels),
                  "measured": gid in measured_gates}
            for gid, scores in sorted(gate_scores.items())
        },
        "checks": {code: flag_rates(fired, labels)
                   for code, fired in sorted(check_fired.items())},
        # A code that fired but is on neither list means risk.py grew a check
        # this harness does not know about — surfaced, never dropped.
        "unclassified_codes": sorted(
            seen_codes - set(REPLAYABLE_CODES) - set(UNMEASURED_CODES)),
        "risk_coverage": risk_coverage(verdict_scores, labels),
        "not_measured": {
            "gates": sorted(set(gate_scores) - measured_gates),
            "checks": unmeasured_codes,
            "reason": "no run in this set exercised them — re-run with "
                      "`bird --trust-eval` for the judge gate; the three "
                      "value-grounded checks additionally need PostgreSQL, "
                      "which the SQLite benchmark does not use",
        },
        "runs_with_stored_verification": sum(
            1 for r in records
            if isinstance(r.get("verification"), dict) and r["verification"]),
    }


# ── report ────────────────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    return "  n/a" if v is None else f"{v:5.1f}"


def print_report(res: Dict[str, Any]) -> None:
    print(f"\nTrust-detector evaluation — {res['total']} labelled records "
          f"from {len(res['runs'])} run(s), {res['correct']} correct "
          f"({100 * res['correct'] / res['total']:.1f}%)")
    print(f"models: {', '.join(res['models'])}\n")

    print("AUROC — probability a correct answer outranks an incorrect one")
    print("  (0.50 = no signal.  Published: self-consistency 0.61–0.68, "
          "LLM judge 0.72–0.78)")
    print(f"\n  {'signal':<24}{'AUROC':>8}")
    for gid, s in res["gates"].items():
        note = "" if s["measured"] else "   (not measured)"
        val = "  n/a" if s["auroc"] is None else f"{s['auroc']:.3f}"
        print(f"  {gid:<24}{val:>8}{note}")
    v = res["verdict"]["auroc"]
    print(f"  {'COMBINED VERDICT':<24}{('  n/a' if v is None else f'{v:.3f}'):>8}")

    print("\nPer-check accusation quality")
    print(f"  {'risk code':<28}{'fired':>6}{'precis.':>9}{'recall':>8}{'false alarm':>13}")
    for name, r in res["checks"].items():
        print(f"  {name:<28}{r['fired']:>6}{_pct(r['precision_pct']):>9}"
              f"{_pct(r['recall_pct']):>8}{_pct(r['false_alarm_pct']):>13}")

    print("\nVerdict bands")
    print(f"  {'verdict':<14}{'n':>6}{'share':>9}{'correct':>10}")
    for lvl, b in res["verdict"]["bands"].items():
        print(f"  {lvl:<14}{b['n']:>6}{_pct(b['share_pct'])}%{_pct(b['accuracy_pct'])}%")

    print("\nRisk–coverage (answer only above the threshold)")
    print(f"  {'threshold':>10}{'coverage':>10}{'selective acc.':>16}")
    for pt in res["risk_coverage"]:
        print(f"  {pt['threshold']:>10.2f}{pt['coverage_pct']:>9.1f}%"
              f"{pt['selective_accuracy_pct']:>15.1f}%")

    if res.get("unclassified_codes"):
        print(f"\n  ⚠ codes this harness does not classify: "
              f"{', '.join(res['unclassified_codes'])}")

    nm = res["not_measured"]
    print(f"\nNOT measured — {nm['reason']}")
    print(f"  gates : {', '.join(nm['gates'])}")
    print(f"  checks: {', '.join(nm['checks'])}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="bird_*.json",
                    help="Glob inside outputs/ (default: bird_*.json)")
    ap.add_argument("--output", default=None,
                    help="Write the full result as JSON to this path")
    args = ap.parse_args()

    records = load_records(args.pattern)
    if not records:
        print(f"No benchmark records matched outputs/{args.pattern}")
        return
    res = evaluate(records)
    print_report(res)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
