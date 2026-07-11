"""
scripts/build_factual_eval.py — Freeze the factual-lookup eval set (v2).

Builds data/eval/factual_queries.jsonl: (query, expected_value, fact_id)
pairs for the deterministic lookup path, scored by exact match in
evals/factual_task.py.

Frozen as a file (rather than generated at eval time) on purpose: an eval
set must not drift silently when the facts DB changes. Rebuilding is an
explicit act with a diff.

Sources:
    - every fact-derived lookup query in data/intent/intent_queries.jsonl
      (both splits — this evaluates the lookup pipeline, not the classifier)
    - additional paraphrases per template family
    - "ambiguous_ok" rows where the correct behavior is a clarification,
      not a value (underspecified efficiency queries)

Usage:
    python scripts/build_factual_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INTENT_DATA = ROOT / "data" / "intent" / "intent_queries.jsonl"
OUT = ROOT / "data" / "eval" / "factual_queries.jsonl"

# Hand-written paraphrase → fact_id (checked against the DB at build time).
_EXTRA = [
    ("How many pounds of anhydrous ammonia trigger PSM coverage?", "osha_psm_tq_ammonia_anhydrous"),
    ("PSM threshold quantity for chlorine gas?", "osha_psm_tq_chlorine"),
    ("What's the TQ for hydrogen sulfide under 1910.119?", "osha_psm_tq_hydrogen_sulfide"),
    ("Threshold quantity of formaldehyde for process safety management?", "osha_psm_tq_formaldehyde"),
    ("How often do PSM compliance audits have to happen?", "osha_psm_compliance_audit"),
    ("Deadline for starting an incident investigation under PSM?", "osha_psm_incident_investigation_start"),
    ("How long are incident investigation reports kept?", "osha_psm_incident_report_retention"),
    ("PHA revalidation interval?", "osha_psm_pha_revalidation"),
    ("Refresher training interval for process operators?", "osha_psm_refresher_training"),
    ("Flammable liquid quantity that makes a process PSM-covered?", "osha_psm_flammable_threshold"),
    ("Minimum HSPF2 for split-system heat pumps?", "doe2017_ss_hp_national_hspf2"),
    ("SEER2 floor for single-package air conditioners?", "doe2017_sp_ac_national_seer2"),
    ("What SEER2 must space-constrained heat pumps meet?", "doe2017_sc_hp_national_seer2"),
    ("Off mode watts allowed for split-system heat pumps?", "doe2017_ss_hp_offmode_watts"),
    ("Which states count as the Southwest under the DOE AC rule?", "doe2017_southwest_region_definition"),
    ("Compliance date for the amended CAC/HP standards?", "doe2017_compliance_date"),
]

# Underspecified queries where the CORRECT outcome is a clarification:
# several facts match equally well and their values differ, so answering
# any single value would be overconfident.
_AMBIGUOUS_OK = [
    "What is the TQ for hydrogen?",           # chloride/fluoride/peroxide/selenide/sulfide
    "Minimum SEER2 for split-system units?",  # AC (13.4) vs heat pump (14.3)
]


def main() -> None:
    from src.facts import FactsDB

    db = FactsDB(ROOT / "data" / "facts")
    by_id = {f.fact_id: f for f in db.facts}

    rows: list[dict] = []
    for line in INTENT_DATA.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("source") == "fact_derived" and r.get("fact_id") in by_id:
            fact = by_id[r["fact_id"]]
            rows.append({
                "query": r["query"],
                "fact_id": fact.fact_id,
                "expected_value": fact.value,
                "expected_page": fact.source_page,
                "expect": "value",
            })

    for query, fact_id in _EXTRA:
        if fact_id not in by_id:
            print(f"WARNING: skipping {query!r} — unknown fact_id {fact_id}")
            continue
        fact = by_id[fact_id]
        rows.append({
            "query": query,
            "fact_id": fact.fact_id,
            "expected_value": fact.value,
            "expected_page": fact.source_page,
            "expect": "value",
        })

    rows.extend(
        {"query": q, "fact_id": None, "expected_value": None,
         "expected_page": None, "expect": "clarification"}
        for q in _AMBIGUOUS_OK
    )

    # De-dup
    seen: set[str] = set()
    unique = []
    for r in rows:
        key = r["query"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    with OUT.open("w") as fh:
        for r in unique:
            fh.write(json.dumps(r) + "\n")
    n_value = sum(1 for r in unique if r["expect"] == "value")
    print(f"Wrote {OUT}: {len(unique)} queries ({n_value} value, "
          f"{len(unique) - n_value} clarification)")


if __name__ == "__main__":
    main()
