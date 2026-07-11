# Corpus Selection

This document explains the criteria, sources, and decisions behind the engineering PDF corpus used by the assistant.

## Selection Criteria

Documents in `data/documents/` are chosen against the following rules:

1. Public or license-compatible. Government publications, NASA Technical Reports, OSHA regulations, and public educational guides are preferred. Paid standards must not be committed.
2. Text-extractable. Native-text PDFs only. Scanned/image PDFs are out of scope until OCR is added.
3. Page-numbered. Page numbering must be stable and meaningful so citations are interpretable in the UI.
4. Explainable. Each document must have a clear purpose in the corpus and a clear retrieval challenge it helps test.
5. Domain spread. The corpus should span 2-3 engineering domains so retrieval is not trivially solved by a single document.

## Target Mix

| Slot | Document type | Why it belongs |
|------|---------------|----------------|
| 1-3 | NASA Technical Reports (https://ntrs.nasa.gov) | Long-form technical writing, named figures and tables, good for testing recall on specific page references. |
| 4-5 | OSHA regulations (https://www.osha.gov/laws-regs/regulations/standardnumber) | Highly structured numeric subsections, exercises exact-term BM25 retrieval (e.g. "1910.119(e)"). |
| 6-7 | Public energy-efficiency guides (DOE, EPA, ASHRAE public summaries) | Numeric tables and unit-sensitive queries (psi, °F, inches). Good for testing whether the answer cites the correct row. |
| 8-10 | Public mechanical/piping educational references | Domain-specific acronyms (NPSH, BHN, CFM). Stresses hybrid retrieval over dense-only. |

## Exclusions

- Paid standards copied into the repo (ASME B16.5 full text, ASHRAE 90.1 full text, API 17D full text, etc.)
- Private employer documents
- Documents with unclear licensing
- Scanned PDFs (until OCR is added)

## Corpus Manifest

After selection, record each document in the table below so reviewers and future contributors understand provenance.

| Filename | Source URL | License/access note | Pages | Why included |
|----------|------------|---------------------|-------|--------------|
| `doe_hdbk_1012_v1_thermodynamics.pdf` | [DOE-HDBK-1012/1-92](https://www.standards.doe.gov/standards-documents/1000/1012-BHdbk-1992-V1) | US government work (public domain) | 139 | Thermodynamics fundamentals (properties, energy balances). Core HVAC physics; long-form educational prose with formulas and tables. |
| `doe_hdbk_1012_v2_heat_transfer.pdf` | [DOE-HDBK-1012/2-92](https://www.standards.doe.gov/standards-documents/1000/1012-bhdbk-1992-v2) | US government work (public domain) | 80 | Heat transfer (conduction, convection, radiation, heat exchangers). Unit-sensitive numeric queries (Btu/hr, °F). |
| `doe_hdbk_1018_v2_mechanical_science.pdf` | [DOE-HDBK-1018/2-93](https://www.standards.doe.gov/standards-documents/1000/1018-bhdbk-1993-v2) | US government work (public domain) | 130 | Valves and miscellaneous mechanical components (valve types, filters, strainers, air compressors). Domain terminology stresses hybrid BM25+dense retrieval. |
| `nasa_systems_engineering_handbook.pdf` | [NASA SP-2016-6105 Rev 2](https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf) | NASA publication (public domain) | 297 | Systems engineering lifecycle, requirements, verification, reviews. Largest document; tests recall across long-form chapters. |
| `osha_3132_process_safety_management.pdf` | [OSHA 3132 (2000, reprinted)](https://www.osha.gov/sites/default/files/publications/OSHA3132.pdf) | US government work (public domain) | 59 | PSM overview with full 29 CFR 1910.119 standard text as appendix. Highly structured numeric subsections exercise exact-term retrieval. |
| `doe_final_rule_2017_cac_hp_efficiency.pdf` | [82 FR 1786, Jan 6, 2017 (doc 2016-29992)](https://www.govinfo.gov/content/pkg/FR-2017-01-06/pdf/2016-29992.pdf) | Federal Register (public domain) | 73 | Energy conservation standards for residential central AC/heat pumps (Jan 1, 2023 compliance). Dense regulatory text in 3-column layout; stresses parsing and table lookup. |

The author worked with these document types across 12 years in HVAC, subsea, and manufacturing engineering. The corpus reflects the problem this assistant addresses: engineering knowledge locked in PDFs. The 2017 DOE final rule is the regulation behind the January 2023 product-line redesign the author led at Rheem Manufacturing.

## Notes

- Update this file whenever a document is added, removed, or replaced.
- The corpus manifest is the source of truth for what the evaluation set should reference.
