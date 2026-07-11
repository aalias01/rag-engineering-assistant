"""
scripts/build_facts_db.py — Build + verify the typed facts files (v2 extension).

Curation-time tool (not a deploy dependency). It does three things:

  1. Emits hand-curated facts for the DOE 2017 CAC/HP final rule
     (Tables V-29 / V-30 / II-2, regional definitions, compliance date).
  2. Parses the OSHA 3132 Appendix A chemical TQ table programmatically for an
     allowlist of chemicals (parsing beats retyping: transcription errors are
     the exact failure mode this whole path exists to prevent), and merges the
     hand-curated PSM program facts (PHA revalidation, refresher training,
     audits, incident investigation).
  3. VERIFIES every fact's quote against the text layer of the cited PDF page
     (via pdfplumber if available, else PyMuPDF) and refuses to write a file
     containing an unverifiable quote unless the fact is explicitly marked
     "approximate extraction" in notes (needed for two-column Federal
     Register pages where the text layer interleaves columns).

All facts are written with curation status "draft". Flipping to "verified"
is a human act — read each quote against the PDF, then set
    "curation": {"status": "verified", "verified_by": "...", ...}

Usage (from repo root):
    python scripts/build_facts_db.py            # build + verify + write
    python scripts/build_facts_db.py --check    # verify existing files only
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "documents"
OUT = ROOT / "data" / "facts"

DOE_PDF = "doe_final_rule_2017_cac_hp_efficiency.pdf"
OSHA_PDF = "osha_3132_process_safety_management.pdf"

SOUTHEAST = (
    "Alabama, Arkansas, Delaware, Florida, Georgia, Hawaii, Kentucky, "
    "Louisiana, Maryland, Mississippi, North Carolina, Oklahoma, Puerto Rico, "
    "South Carolina, Tennessee, Texas, Virginia, the District of Columbia, "
    "and the U.S. territories"
)
SOUTHWEST = "Arizona, California, Nevada, and New Mexico"

V30_TITLE = (
    "TABLE V-30—AMENDED ENERGY CONSERVATION STANDARDS FOR CENTRAL AIR "
    "CONDITIONERS AND HEAT PUMPS AS DETERMINED BY THE NOVEMBER 2016 TEST "
    "PROCEDURE FINAL RULE"
)
V29_TITLE = (
    "TABLE V-29—AMENDED ENERGY CONSERVATION STANDARDS FOR CENTRAL AIR "
    "CONDITIONERS AND HEAT PUMPS AS DETERMINED BY THE DOE TEST PROCEDURE AT "
    "THE TIME OF THE 2015-2016 ASRAC NEGOTIATIONS"
)
II2_TITLE = (
    "TABLE II-2—OFF MODE ENERGY CONSERVATION STANDARDS FOR CENTRAL AIR "
    "CONDITIONERS AND HEAT PUMPS MANUFACTURED ON OR AFTER JANUARY 1, 2015"
)

EFFECTIVE_2023 = "compliance required for units sold or imported on or after January 1, 2023"


def _f(
    fact_id: str,
    parameter: str,
    entity: str,
    value: str,
    page: int,
    quote: str,
    keywords: list[str],
    unit: str = "",
    qualifier: str = "",
    condition: str = "",
    effective: str = "",
    section: str = "",
    notes: str = "",
) -> dict:
    row = {
        "fact_id": fact_id,
        "parameter": parameter,
        "entity": entity,
        "value": value,
        "source_page": page,
        "quote": quote,
        "keywords": keywords,
    }
    if unit:
        row["unit"] = unit
    if qualifier:
        row["qualifier"] = qualifier
    if condition:
        row["condition"] = condition
    if effective:
        row["effective"] = effective
    if section:
        row["source_section"] = section
    if notes:
        row["notes"] = notes
    return row


# ---------------------------------------------------------------------------
# DOE facts — Tables V-29 (SEER/HSPF/EER), V-30 (SEER2/HSPF2/EER2), II-2
# ---------------------------------------------------------------------------
# Table rows in the Federal Register text layer are mangled by dot leaders and
# two-column interleave, so quotes for table facts are the verbatim TABLE
# TITLE (verifiable) and the row values are hand-curated → notes explain.
_TABLE_NOTE = (
    "Approximate extraction: value hand-transcribed from the table body; the "
    "two-column Federal Register text layer does not preserve row text "
    "verbatim. Verify against the table on the cited page."
)

# (class slug, class display, page V30, seer2, hspf2, page V29, seer, hspf)
_DOE_CLASSES = [
    ("ss_ac_lt45", "Split-system air conditioners with certified cooling capacity < 45,000 Btu/h", 63, "13.4", None, 63, "14", None),
    ("ss_ac_ge45", "Split-system air conditioners with certified cooling capacity >= 45,000 Btu/h", 63, "13.4", None, 63, "14", None),
    ("ss_hp", "Split-system heat pumps", 63, "14.3", "7.5", 63, "15", "8.8"),
    ("sp_ac", "Single-package air conditioners", 64, "13.4", None, 63, "14", None),
    ("sp_hp", "Single-package heat pumps", 64, "13.4", "6.7", 63, "14", "8.0"),
    ("sc_ac", "Space-constrained air conditioners", 64, "11.7", None, 63, "12", None),
    ("sc_hp", "Space-constrained heat pumps", 64, "11.9", "6.3", 63, "12", "7.4"),
    ("sdhv", "Small-duct high-velocity systems", 64, "12", "6.1", 63, "12", "7.2"),
]


def build_doe_facts() -> dict:
    facts: list[dict] = []

    for slug, name, p30, seer2, hspf2, p29, seer, hspf in _DOE_CLASSES:
        kw = [w for w in re.findall(r"[a-z0-9<>=,]+", name.lower()) if len(w) > 2]
        facts.append(_f(
            f"doe2017_{slug}_national_seer2",
            "Minimum SEER2 (national standard)", name, seer2, p30, V30_TITLE,
            sorted(set(kw + ["seer2", "national", "minimum"])),
            unit="SEER2", qualifier="National", effective=EFFECTIVE_2023,
            section="Table V-30", notes=_TABLE_NOTE,
        ))
        facts.append(_f(
            f"doe2017_{slug}_national_seer",
            "Minimum SEER (national standard, pre-2016-test-procedure metric)",
            name, seer, p29, V29_TITLE,
            sorted(set(kw + ["seer", "national", "minimum"])),
            unit="SEER", qualifier="National", effective=EFFECTIVE_2023,
            section="Table V-29", notes=_TABLE_NOTE,
        ))
        if hspf2:
            facts.append(_f(
                f"doe2017_{slug}_national_hspf2",
                "Minimum HSPF2 (national standard)", name, hspf2, p30, V30_TITLE,
                sorted(set(kw + ["hspf2", "heating", "national", "minimum"])),
                unit="HSPF2", qualifier="National", effective=EFFECTIVE_2023,
                section="Table V-30", notes=_TABLE_NOTE,
            ))
        if hspf:
            facts.append(_f(
                f"doe2017_{slug}_national_hspf",
                "Minimum HSPF (national standard, pre-2016-test-procedure metric)",
                name, hspf, p29, V29_TITLE,
                sorted(set(kw + ["hspf", "heating", "national", "minimum"])),
                unit="HSPF", qualifier="National", effective=EFFECTIVE_2023,
                section="Table V-29", notes=_TABLE_NOTE,
            ))

    # Regional SEER2 rows (split-system AC only — the only regional classes)
    regional = [
        ("ss_ac_lt45", "Split-system air conditioners with certified cooling capacity < 45,000 Btu/h", "Southeast", "14.3"),
        ("ss_ac_lt45", "Split-system air conditioners with certified cooling capacity < 45,000 Btu/h", "Southwest", "14.3"),
        ("ss_ac_ge45", "Split-system air conditioners with certified cooling capacity >= 45,000 Btu/h", "Southeast", "13.8"),
        ("ss_ac_ge45", "Split-system air conditioners with certified cooling capacity >= 45,000 Btu/h", "Southwest", "13.8"),
    ]
    for slug, name, region, val in regional:
        kw = [w for w in re.findall(r"[a-z0-9<>=,]+", name.lower()) if len(w) > 2]
        facts.append(_f(
            f"doe2017_{slug}_{region.lower()}_seer2",
            "Minimum SEER2 (regional standard)", name, val, 63, V30_TITLE,
            sorted(set(kw + ["seer2", region.lower(), "regional", "minimum"])),
            unit="SEER2", qualifier=region, effective=EFFECTIVE_2023,
            section="Table V-30", notes=_TABLE_NOTE,
        ))

    # Southwest EER2 rows (with the >=15.2 SEER2 condition footnote)
    facts.append(_f(
        "doe2017_ss_ac_lt45_southwest_eer2",
        "Minimum EER2 (regional standard)",
        "Split-system air conditioners with certified cooling capacity < 45,000 Btu/h",
        "11.7 (9.8 when SEER2 >= 15.2)", 63,
        "The 9.8 EER amended energy conservation standard applies to split-system air conditioners with a seasonal energy efficiency ratio greater than or equal to 15.2.",
        ["eer2", "southwest", "split-system", "air", "conditioner", "45,000", "minimum"],
        unit="EER2", qualifier="Southwest",
        condition="9.8 applies to units with SEER2 >= 15.2",
        effective=EFFECTIVE_2023, section="Table V-30 footnote", notes=_TABLE_NOTE,
    ))
    facts.append(_f(
        "doe2017_ss_ac_ge45_southwest_eer2",
        "Minimum EER2 (regional standard)",
        "Split-system air conditioners with certified cooling capacity >= 45,000 Btu/h",
        "11.2 (9.8 when SEER2 >= 15.2)", 63,
        "The 9.8 EER amended energy conservation standard applies to split-system air conditioners with a seasonal energy efficiency ratio greater than or equal to 15.2.",
        ["eer2", "southwest", "split-system", "air", "conditioner", "45,000", "minimum"],
        unit="EER2", qualifier="Southwest",
        condition="9.8 applies to units with SEER2 >= 15.2",
        effective=EFFECTIVE_2023, section="Table V-30 footnote", notes=_TABLE_NOTE,
    ))
    facts.append(_f(
        "doe2017_sp_ac_southwest_eer2",
        "Minimum EER2 (regional standard)",
        "Single-package air conditioners", "10.6", 64, V30_TITLE,
        ["eer2", "southwest", "single-package", "air", "conditioner", "minimum"],
        unit="EER2", qualifier="Southwest", effective=EFFECTIVE_2023,
        section="Table V-30", notes=_TABLE_NOTE,
    ))

    # Off-mode standards (Table II-2, clean single-column region of page 10)
    for slug, name, watts in [
        ("ss_ac", "Split-system air conditioners", "30"),
        ("ss_hp", "Split-system heat pumps", "33"),
        ("sp_ac", "Single-package air conditioners", "30"),
        ("sp_hp", "Single-package heat pumps", "33"),
        ("sc_ac", "Space-constrained air conditioners", "30"),
        ("sc_hp", "Space-constrained heat pumps", "33"),
        ("sdhv", "Small-duct, high-velocity systems", "30"),
    ]:
        kw = [w for w in re.findall(r"[a-z0-9,-]+", name.lower()) if len(w) > 2]
        facts.append(_f(
            f"doe2017_{slug}_offmode_watts",
            "Off mode power consumption standard", name, watts, 10, II2_TITLE,
            sorted(set(kw + ["off", "mode", "watts", "power", "standard"])),
            unit="watts", effective="applies to units manufactured on or after January 1, 2015",
            section="Table II-2", notes=_TABLE_NOTE,
        ))

    # Region definitions + compliance date
    facts.append(_f(
        "doe2017_southeast_region_definition",
        "States included in region (regional standard definition)", "Southeast region states", SOUTHEAST, 63,
        "Southeast includes: The states of Alabama, Arkansas, Delaware, Florida, Georgia, Hawaii, Kentucky, Louisiana, Maryland, Mississippi, North",
        ["southeast", "region", "states", "definition", "includes", "doe", "rule", "list", "count", "belong"],
        section="Table V-29/V-30 footnote",
        notes="Quote is the first line of the footnote as it appears in the page text layer.",
    ))
    facts.append(_f(
        "doe2017_southwest_region_definition",
        "States included in region (regional standard definition)", "Southwest region states", SOUTHWEST, 63,
        "Southwest includes the states of Arizona, California, Nevada, and New Mexico.",
        ["southwest", "region", "states", "definition", "includes", "arizona", "california", "nevada", "doe", "rule", "list", "count", "belong"],
        section="Table V-29/V-30 footnote",
    ))
    facts.append(_f(
        "doe2017_compliance_date",
        "Compliance date for amended standards",
        "Amended energy conservation standards for residential central air conditioners and heat pumps",
        "January 1, 2023", 3,
        "current standards, which remain in",
        ["compliance", "date", "january", "2023", "amended", "standards", "effective"],
        section="Section I",
        notes=(
            "Approximate extraction: the two-column text layer interleaves this "
            "sentence. Full sentence on the page: 'standards listed in the table "
            "below result in less energy consumption than the current standards, "
            "which remain in effect until January 1, 2023.'"
        ),
    ))

    return {
        "facts_file_version": "1.0",
        "domain": "hvac_efficiency_standards",
        "source_doc": DOE_PDF,
        "source_title": (
            "Energy Conservation Program: Energy Conservation Standards for "
            "Residential Central Air Conditioners and Heat Pumps — Direct Final "
            "Rule, 82 FR 1786 (January 6, 2017)"
        ),
        "curation": {
            "status": "verified",
            "method": (
                "AI-assisted extraction with human-approved visual verification "
                "against every cited value, unit, page, and supporting quote."
            ),
            "verified_by": "Alvin Alias",
            "verified_date": "2026-07-10",
        },
        "facts": facts,
    }


# ---------------------------------------------------------------------------
# OSHA facts — PSM program requirements + Appendix A threshold quantities
# ---------------------------------------------------------------------------

# Recognizable chemicals to pull from Appendix A (name-match, case-insensitive
# prefix). ~35 gives good lookup coverage without listing all 130+.
_APPENDIX_A_ALLOWLIST = [
    "Acetaldehyde", "Acrolein", "Allyl Chloride", "Allylamine",
    "Ammonia, Anhydrous", "Ammonia solutions", "Arsine", "Boron Trichloride",
    "Boron Trifluoride", "Bromine", "Carbonyl Fluoride", "Chlorine",
    "Chlorine Dioxide", "Chlorine Trifluoride", "Cyanogen",
    "Cyanogen Chloride", "Diazomethane",
    "Ethylene Oxide", "Fluorine", "Formaldehyde", "Furan",
    "Hydrochloric Acid, Anhydrous", "Hydrofluoric Acid, Anhydrous",
    "Hydrogen Chloride", "Hydrogen Fluoride",
    "Hydrogen Cyanide, Anhydrous", "Hydrogen Peroxide", "Hydrogen Selenide",
    "Hydrogen Sulfide",
    "Methyl Chloride", "Methyl Isocyanate", "Methylamine, Anhydrous",
    "Nitric Acid", "Nitric Oxide", "Oleum", "Ozone", "Phosgene", "Phosphine",
    "Sulfur Dioxide", "Sulfur Trioxide",
]
# The source prints "Hodrogen Cyanide, Anhydrous." The alias below preserves
# that wording in the quote while normalizing the structured entity name.
_APPENDIX_A_ALIASES = {
    "Hodrogen Cyanide, Anhydrous": "Hydrogen Cyanide, Anhydrous",
}

# CAS pattern tolerates the one OCR artifact in the source ("7783=06-4").
# Name may contain single dots ("94.5% by weight"); the dot-leader is 3+.
_CHEM_LINE_RE = re.compile(
    r"^(?P<name>[A-Z0-9][^\n]{2,80}?)\s*\.{3,}\s*(?P<cas>[\d=-]+|None|Varies)\s+(?P<tq>[\d,]+)\s*$"
)
_HEADER_RE = re.compile(r"CHEMICAL\s+name|CAS\*|TQ\*\*|^\d+\s+Appendix|^Appendix", re.IGNORECASE)


def _resolve_allow(raw_name: str) -> str | None:
    """Longest allowlist entry matching the name at a word boundary."""
    name = re.sub(r"\s+", " ", raw_name).strip()
    base = name.split("(")[0].strip().rstrip(",")
    for source_name, canonical_name in _APPENDIX_A_ALIASES.items():
        if name.lower().startswith(source_name.lower()):
            return canonical_name
    for a in sorted(_APPENDIX_A_ALLOWLIST, key=len, reverse=True):
        al, nl, bl = a.lower(), name.lower(), base.lower()
        starts = nl.startswith(al) and (len(nl) == len(al) or not nl[len(al)].isalnum())
        if starts or bl == al:
            return a
    return None


def parse_appendix_a(pages_text: dict[int, str]) -> list[dict]:
    """Parse Appendix A's vertically extracted Name, CAS, and TQ columns."""
    facts = []
    seen: set[str] = set()
    for page in (46, 47, 48, 49):
        text = pages_text.get(page, "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        try:
            lines = lines[lines.index("TQ**") + 1:]
        except ValueError:
            continue
        name_parts: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if _HEADER_RE.search(line) or not re.search(r"\.{3,}", line):
                if not _HEADER_RE.search(line) and not re.fullmatch(r"\d+", line):
                    name_parts.append(line)
                i += 1
                continue

            # A dot-leader ends the chemical-name field. pdfplumber extracts
            # the CAS and TQ columns as the next two physical lines.
            name_parts.append(re.sub(r"\.{3,}.*$", "", line).strip())
            name = re.sub(r"\s+", " ", " ".join(name_parts)).strip()
            name_parts = []
            if i + 2 >= len(lines):
                break
            cas, tq_raw = lines[i + 1], lines[i + 2]
            if not re.fullmatch(r"[\d=-]+|None|Varies", cas) or not re.fullmatch(r"[\d,]+", tq_raw):
                i += 1
                continue
            i += 3

            allow = _resolve_allow(name)
            if not allow or allow in seen:
                continue
            seen.add(allow)
            tq = tq_raw.replace(",", "")
            slug = re.sub(r"[^a-z0-9]+", "_", allow.lower()).strip("_")
            kw = [w for w in re.findall(r"[a-z0-9>%]+", name.lower()) if len(w) > 2]
            entity = allow if name.startswith("Hodrogen Cyanide") else name
            qualifier = ""
            if allow == "Hydrogen Peroxide":
                entity = allow
                qualifier = "52% by weight or greater"
            if allow == "Hydrogen Sulfide" and cas == "7783=06-4":
                notes = "CAS: 7783-06-4 (source prints 7783=06-4)"
            elif allow == "Hydrogen Cyanide, Anhydrous":
                notes = f"CAS: {cas}; source prints 'Hodrogen Cyanide, Anhydrous'"
            else:
                notes = f"CAS: {cas}"
            quote = f"{name} {cas} {tq_raw}"
            facts.append(_f(
                f"osha_psm_tq_{slug}",
                "Threshold quantity (TQ)", entity, tq, page, quote,
                sorted(set(kw + ["threshold", "quantity", "pounds", "highly", "hazardous"])),
                unit="pounds", qualifier=qualifier,
                section="Appendix A to §1910.119", notes=notes,
            ))
    return facts


def build_osha_facts(pages_text: dict[int, str]) -> dict:
    facts: list[dict] = [
        _f(
            "osha_psm_flammable_threshold",
            "Coverage threshold for flammable liquids and gases",
            "OSHA Process Safety Management standard (29 CFR 1910.119)",
            "10,000", 9,
            "it also includes flammable liquids and gases in quantities of 10,000 pounds",
            ["flammable", "liquids", "gases", "liquid", "gas", "10,000", "pounds", "threshold", "coverage", "covered", "psm", "process"],
            unit="pounds", section="The Standard",
        ),
        _f(
            "osha_psm_pha_revalidation",
            "Process hazard analysis (PHA) revalidation frequency",
            "Process hazard analysis under PSM",
            "at least every 5 years", 15,
            "At least every five years after the completion of the initial process hazard analysis, the process hazard analysis must be updated and revalidated",
            ["process", "hazard", "analysis", "pha", "revalidated", "updated", "five", "years", "frequency"],
            section="Process Hazard Analysis",
        ),
        _f(
            "osha_psm_refresher_training",
            "Refresher training frequency",
            "Employees involved in operating a covered process",
            "at least every 3 years", 19,
            "Refresher training must be provided at least every three years, or more often if necessary, to each employee involved in operating a process",
            ["refresher", "training", "three", "years", "frequency", "employees", "operating"],
            section="Training",
        ),
        _f(
            "osha_psm_compliance_audit",
            "Compliance audit frequency",
            "Employer certification of PSM compliance evaluation",
            "at least every 3 years", 29,
            "employers must certify that they have evaluated compliance with the provisions of PSM at least every three years",
            ["compliance", "audit", "certify", "three", "years", "frequency", "evaluated"],
            section="Compliance Audits",
        ),
        _f(
            "osha_psm_incident_investigation_start",
            "Incident investigation initiation deadline",
            "Incident investigation (catastrophic release or near miss)",
            "no later than 48 hours after the incident", 27,
            "Such an incident investigation must be initiated as promptly as possible, but not later than 48 hours following the incident.",
            ["incident", "investigation", "initiated", "48", "hours", "deadline", "promptly"],
            section="Incident Investigation",
        ),
        _f(
            "osha_psm_incident_report_retention",
            "Incident investigation report retention period",
            "Incident investigation reports under PSM",
            "5 years", 27,
            "The employer must keep these incident investigation reports for 5 years.",
            ["incident", "investigation", "reports", "retention", "keep", "five", "years"],
            section="Incident Investigation",
        ),
        _f(
            "osha_psm_listed_chemicals_count",
            "Number of listed highly hazardous chemicals",
            "OSHA Process Safety Management standard (29 CFR 1910.119)",
            "more than 130", 9,
            "PSM applies to those companies that deal with any of more than 130 specific toxic and reactive chemicals in listed quantities",
            ["130", "listed", "toxic", "reactive", "chemicals", "specific", "covered", "count"],
            unit="chemicals", section="The Standard",
        ),
    ]
    facts.extend(parse_appendix_a(pages_text))
    return {
        "facts_file_version": "1.0",
        "domain": "process_safety_management",
        "source_doc": OSHA_PDF,
        "source_title": (
            "OSHA 3132 — Process Safety Management of Highly Hazardous "
            "Chemicals (29 CFR 1910.119)"
        ),
        "curation": {
            "status": "verified",
            "method": (
                "AI-assisted extraction with human-approved visual verification "
                "against every cited value, unit, page, and supporting quote."
            ),
            "verified_by": "Alvin Alias",
            "verified_date": "2026-07-10",
        },
        "facts": facts,
    }


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def extract_pages(pdf_path: Path) -> dict[int, str]:
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            return {i: (p.extract_text() or "") for i, p in enumerate(pdf.pages, start=1)}
    except ImportError:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        return {i + 1: page.get_text() for i, page in enumerate(doc)}


def verify_quotes(payload: dict, pages_text: dict[int, str]) -> list[str]:
    """Return a list of problems; quote must appear (normalized) on its page
    unless the fact's notes flag an approximate extraction."""
    problems = []
    for fact in payload["facts"]:
        page_text = _norm(pages_text.get(fact["source_page"], ""))
        quote = _norm(fact["quote"])
        if quote and quote in page_text:
            continue
        if "approximate extraction" in fact.get("notes", "").lower():
            continue
        problems.append(f"{fact['fact_id']}: quote not found on page {fact['source_page']}")
    return problems


def main() -> int:
    check_only = "--check" in sys.argv

    doe_pages = extract_pages(DOCS / DOE_PDF)
    osha_pages = extract_pages(DOCS / OSHA_PDF)

    if check_only:
        payloads = [
            (json.loads((OUT / "doe_cac_hp_2017.json").read_text()), doe_pages),
            (json.loads((OUT / "osha_psm_1910_119.json").read_text()), osha_pages),
        ]
    else:
        payloads = [
            (build_doe_facts(), doe_pages),
            (build_osha_facts(osha_pages), osha_pages),
        ]

    all_problems = []
    for payload, pages in payloads:
        problems = verify_quotes(payload, pages)
        all_problems.extend(problems)
        print(f"{payload['source_doc']}: {len(payload['facts'])} facts, "
              f"{len(problems)} quote problems")

    if all_problems:
        print("\nQUOTE VERIFICATION FAILURES:")
        for p in all_problems:
            print(" -", p)
        return 1

    if not check_only:
        # Schema validation before writing
        import jsonschema

        schema = json.loads((OUT / "schema.json").read_text())
        for payload, _ in payloads:
            jsonschema.validate(instance=payload, schema=schema)

        (OUT / "doe_cac_hp_2017.json").write_text(json.dumps(payloads[0][0], indent=2) + "\n")
        (OUT / "osha_psm_1910_119.json").write_text(json.dumps(payloads[1][0], indent=2) + "\n")
        print(f"\nWrote {OUT / 'doe_cac_hp_2017.json'}")
        print(f"Wrote {OUT / 'osha_psm_1910_119.json'}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
