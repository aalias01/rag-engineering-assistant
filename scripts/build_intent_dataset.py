"""
scripts/build_intent_dataset.py — Build the labeled intent dataset (v2).

Writes data/intent/intent_queries.jsonl (~210 rows) with a stratified
train/holdout split (fixed seed). Three sources, all transparent:

    fact_derived — lookup queries composed from the facts DB via varied
                   templates (varied on purpose: a classifier that only sees
                   one template learns the template, not the intent)
    hand_written — paraphrases, conceptual questions, and underspecified
                   queries written for this dataset
    eval_set     — the 31 existing hand-labeled retrieval-eval queries,
                   re-labeled for *intent* (they were labeled for retrieval)

LABELING POLICY (also in data/intent/README.md):
    1. "lookup" = the answer is a single stored value (number, date,
       frequency, threshold) retrievable without synthesis.
    2. "interpret" wins whenever reasoning/explanation is requested — even
       about a lookup-able value ("Why is the TQ for phosgene so low?").
       Out-of-corpus questions are also "interpret": routing happens before
       retrieval, and refusal is the interpret path's job.
    3. "clarify" = a competent human expert would have to ask a question
       back before answering ("Minimum efficiency?" — of what, under which
       standard?).

Curation status: REVIEWED. Every label was checked against the policy on
2026-07-10. Five generated labels were corrected during that pass.

Usage:
    python scripts/build_intent_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "intent"
EVAL_QUERIES = ROOT / "data" / "eval" / "test_queries.jsonl"

SEED = 558  # BIOST 558 — fixed for reproducible splits
HOLDOUT_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Lookup — fact-derived templates (filled from the facts DB)
# ---------------------------------------------------------------------------

_TQ_TEMPLATES = [
    "What is the TQ for {chem}?",
    "What is the threshold quantity for {chem}?",
    "How many pounds of {chem} put a process under PSM coverage?",
    "{chem} threshold quantity in pounds?",
    "At what quantity does {chem} become PSM-covered?",
]

_SEER_TEMPLATES = [
    "What is the minimum {metric} for {cls}?",
    "Minimum {metric} for {cls}?",
    "What {metric} do {cls} have to meet?",
    "What is the required {metric} for {cls} in the {region} region?",
    "What is the minimum {metric} for {cls} in the {region}?",
]

_MISC_LOOKUP = [
    "How often must a process hazard analysis be revalidated?",
    "How often is PHA revalidation required under PSM?",
    "What is the refresher training frequency under the PSM standard?",
    "How often must employers provide refresher training to process operators?",
    "How often must PSM compliance audits be performed?",
    "What is the compliance audit frequency under 1910.119?",
    "How soon must an incident investigation be initiated after an incident?",
    "What is the deadline to start an incident investigation under PSM?",
    "How long must incident investigation reports be retained?",
    "What is the retention period for PSM incident investigation reports?",
    "What quantity of flammable liquids brings a process under PSM?",
    "What is the PSM threshold for flammable gases in pounds?",
    "How many chemicals are on the PSM highly hazardous chemicals list?",
    "When do the amended central air conditioner standards take effect?",
    "What is the compliance date for the 2017 CAC/HP efficiency standards?",
    "Which states are in the Southeast region for AC standards?",
    "Which states make up the Southwest region under the DOE rule?",
    "What is the off mode power limit for split-system air conditioners?",
    "What is the off mode standard for single-package heat pumps in watts?",
    "What is the off mode wattage standard for small-duct high-velocity systems?",
]

# ---------------------------------------------------------------------------
# Interpret — hand-written conceptual/reasoning queries across the corpus
# ---------------------------------------------------------------------------

_HAND_INTERPRET = [
    # reasoning about lookup-able values → interpret per policy rule 2
    "Why is the threshold quantity for phosgene so much lower than for ammonia?",
    "Why does the Southwest region have a separate EER2 requirement for air conditioners?",
    "Explain why regional standards exist for central air conditioners.",
    "Why did DOE translate SEER values into SEER2 values in the 2017 rule?",
    "Why does refresher training frequency depend on employee consultation?",
    "Explain the purpose of the PSM compliance audit requirement.",
    "Why must incident investigations start within 48 hours?",
    "How does the PSM standard define a catastrophic release?",
    "What is the purpose of a process hazard analysis?",
    "Explain what management of change means under PSM.",
    "What should an emergency action plan under PSM contain?",
    "How do employers demonstrate mechanical integrity under PSM?",
    "What role do contractors play in process safety management?",
    "Explain the employee participation requirements of the PSM standard.",
    "What is a hot work permit and when is one required?",
    # thermodynamics / heat transfer / mechanical science handbooks
    "Explain the difference between heat and work in thermodynamics.",
    "How does a Carnot cycle establish the maximum possible efficiency?",
    "Why is entropy described as a measure of unavailable energy?",
    "Explain the difference between conduction, convection, and radiation.",
    "How does film boiling differ from nucleate boiling?",
    "Why do heat exchangers use counterflow arrangements?",
    "Explain how a centrifugal pump develops head.",
    "What causes cavitation in pumps and why is it damaging?",
    "Explain the difference between a relief valve and a safety valve.",
    "How does a check valve prevent reverse flow?",
    "Why are expansion joints used in piping systems?",
    "Explain what net positive suction head means.",
    "How does superheating steam improve turbine performance?",
    "Why is subcooling used in refrigeration cycles?",
    "Explain the first law of thermodynamics for open systems.",
    # NASA SE handbook
    "How does NASA define verification versus validation?",
    "Explain the purpose of a preliminary design review.",
    "What does technical risk management involve in NASA's SE process?",
    "How are stakeholder expectations captured in NASA systems engineering?",
    "Explain configuration management in the NASA systems engineering handbook.",
    "What is the role of trade studies in systems design?",
    # comparisons and multi-hop
    "Compare the efficiency metrics SEER2 and EER2 — what does each measure?",
    "How do the DOE efficiency standards differ between split-system and single-package units?",
    "Compare how OSHA treats toxic chemicals versus flammable liquids for PSM coverage.",
    "What is the relationship between HSPF2 and heating performance?",
    # out-of-corpus (routing-wise still interpret; the interpret path refuses)
    "What does ASHRAE 90.1 require for building envelope insulation?",
    "What are the ASME B31.3 requirements for process piping welds?",
    "How does the EPA regulate refrigerant phase-outs under the AIM Act?",
    "What SEER rating does ENERGY STAR require for certification?",
    "What did the 2023 DOE furnace rule change for gas furnaces?",
]

# ---------------------------------------------------------------------------
# Clarify — underspecified queries a competent expert would bounce back
# ---------------------------------------------------------------------------

_HAND_CLARIFY = [
    "What is the minimum efficiency?",
    "What's the minimum SEER2?",
    "What is the threshold quantity?",
    "What's the TQ?",
    "What is the limit?",
    "What are the requirements?",
    "How often is training required?",
    "What is the minimum?",
    "What's the standard say about heat pumps?",
    "What are the rules for air conditioners?",
    "Is 1,500 pounds over the threshold?",
    "Does 12 SEER2 comply?",
    "What about heat pumps?",
    "And for the Southwest?",
    "What's the frequency for that?",
    "How long do we have to keep them?",
    "Is that within the limit?",
    "What is the required value?",
    "Which region applies?",
    "What do I need to comply with?",
    "Tell me the requirements for compliance.",
    "What's the deadline?",
    "How many pounds is the cutoff?",
    "What's the wattage limit?",
    "What efficiency do we need to hit?",
    "Are we covered by the standard?",
    "What does the rule require?",
    "When does it take effect?",
    "What's the revalidation schedule?",
    "How often do we audit?",
    "What are the training rules?",
    "Which chemicals count?",
    "Is chlorine covered?",          # missing: covered by what, at what quantity on site?
    "Do the standards apply to us?",
    "What's the minimum for the North?",   # no "North" region exists in the rule
    "What page is the table on?",
    "Can you check the value again?",
    "What was that number?",
    "Give me the specs.",
    "What's required for the audit?",
]


def _fact_derived_lookups() -> list[dict]:
    import sys

    sys.path.insert(0, str(ROOT))
    from src.facts import FactsDB

    rng = random.Random(SEED)
    db = FactsDB(ROOT / "data" / "facts")
    rows: list[dict] = []

    # TQ queries — sample chemicals, rotate templates
    tq_facts = [f for f in db.facts if f.fact_id.startswith("osha_psm_tq_")]
    for i, fact in enumerate(rng.sample(tq_facts, min(18, len(tq_facts)))):
        chem = fact.entity.split("(")[0].strip().rstrip(",")
        template = _TQ_TEMPLATES[i % len(_TQ_TEMPLATES)]
        rows.append({"query": template.format(chem=chem), "intent": "lookup",
                     "source": "fact_derived", "fact_id": fact.fact_id})

    # Efficiency queries — entity + metric (+ region where the fact has one);
    # only fully-specified combinations (underspecified ones live in clarify).
    metric_by_unit = {"SEER2": "SEER2", "HSPF2": "HSPF2", "EER2": "EER2",
                      "SEER": "SEER", "HSPF": "HSPF"}
    seer_facts = [f for f in db.facts
                  if f.unit in metric_by_unit and "heat pumps" in f.entity.lower()
                  or f.unit in metric_by_unit and "small-duct" in f.entity.lower()
                  or f.unit in metric_by_unit and "space-constrained" in f.entity.lower()
                  or (f.unit in metric_by_unit and f.qualifier in ("Southeast", "Southwest"))]
    for i, fact in enumerate(rng.sample(seer_facts, min(14, len(seer_facts)))):
        cls = " ".join(
            fact.entity.lower().replace(">=", "at least").replace("<", "under ").split()
        )
        region = fact.qualifier if fact.qualifier in ("Southeast", "Southwest") else None
        # Region-qualified facts MUST get a region template (a region-free
        # question about a regionally-varying value is genuinely ambiguous —
        # that's clarify territory, not a lookup with this fact as target).
        if region:
            template = _SEER_TEMPLATES[3 + (i % 2)]  # the two {region} variants
        else:
            template = _SEER_TEMPLATES[i % 3]        # the region-free variants
        rows.append({
            "query": template.format(metric=fact.unit, cls=cls, region=region or "National"),
            "intent": "lookup", "source": "fact_derived", "fact_id": fact.fact_id,
        })
    return rows


def _eval_set_rows() -> list[dict]:
    """Re-label the 31 retrieval-eval queries for intent."""
    rows = []
    # Human-reviewed against the labeling policy on 2026-07-10. Keep exact
    # query text here instead of using lexical markers: the eval set includes
    # single-value questions that are out of corpus and must route to interpret.
    lookup_queries = {
        "What is the value of the Stefan-Boltzmann constant in English units?",
        "How often must a process hazard analysis be updated and revalidated under the PSM standard?",
        "What SEER level did the 2017 DOE final rule set for split-system air conditioners under 45,000 Btu/h in the Southeast region?",
        "When did compliance with the amended energy conservation standards for residential central air conditioners become required?",
    }
    for line in EVAL_QUERIES.read_text().splitlines():
        if not line.strip():
            continue
        q = json.loads(line)["query"]
        intent = "lookup" if q in lookup_queries else "interpret"
        rows.append({"query": q, "intent": intent, "source": "eval_set"})
    return rows


def main() -> None:
    rows: list[dict] = []
    rows += _fact_derived_lookups()
    rows += [{"query": q, "intent": "lookup", "source": "hand_written"} for q in _MISC_LOOKUP]
    rows += [{"query": q, "intent": "interpret", "source": "hand_written"} for q in _HAND_INTERPRET]
    rows += [{"query": q, "intent": "clarify", "source": "hand_written"} for q in _HAND_CLARIFY]
    rows += _eval_set_rows()

    # De-dup on query text
    seen: set[str] = set()
    unique_rows = []
    for row in rows:
        key = row["query"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    # Stratified split
    rng = random.Random(SEED)
    by_intent: dict[str, list[dict]] = {}
    for row in unique_rows:
        by_intent.setdefault(row["intent"], []).append(row)
    final: list[dict] = []
    for intent, group in sorted(by_intent.items()):
        rng.shuffle(group)
        n_holdout = max(1, round(len(group) * HOLDOUT_FRACTION))
        for i, row in enumerate(group):
            row["split"] = "holdout" if i < n_holdout else "train"
            final.append(row)
    rng.shuffle(final)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "intent_queries.jsonl"
    with out_path.open("w") as fh:
        for row in final:
            fh.write(json.dumps(row) + "\n")

    counts = {i: len(g) for i, g in sorted(by_intent.items())}
    holdout = sum(1 for r in final if r["split"] == "holdout")
    print(f"Wrote {out_path}: {len(final)} queries {counts}, holdout={holdout}")


if __name__ == "__main__":
    main()
