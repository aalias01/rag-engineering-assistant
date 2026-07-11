# Typed facts files

Each JSON file here holds atomic, citable facts extracted from one corpus document. The lookup path answers factual questions from these files directly. No language model touches the values.

| File | Source document | Facts |
|---|---|---:|
| `doe_cac_hp_2017.json` | DOE Final Rule 82 FR 1786 (2017 CAC/HP efficiency standards) | 39 |
| `osha_psm_1910_119.json` | OSHA 3132, Process Safety Management | 46 |

Every fact stores the value, unit, source page, and the verbatim sentence it came from. `schema.json` defines the format and `src/facts.py` validates each file against it at load.

## Curation status

Both files are `draft`: values were extracted with machine assistance and verified programmatically against the page text, but the hand-check against the PDFs is not done yet. The API labels every answer built from a draft fact. Flip `curation.status` to `verified` only after reading each fact against its cited page.

## Rebuilding

```bash
python scripts/build_facts_db.py          # rebuild from the PDFs, verify quotes, write
python scripts/build_facts_db.py --check  # verify the existing files only
```

The build fails if any quote cannot be found on its cited page, unless the fact's notes say the extraction is approximate (needed for two-column Federal Register pages, where the PDF text layer interleaves the columns).

## Adding a domain

The schema has no HVAC or safety assumptions. A new facts file needs `domain`, `source_doc` (matching a PDF in `data/documents/`), and facts with `parameter`, `entity`, `value`, `source_page`, `quote`, and `keywords`. Drop it in this directory and the router picks it up on the next API start.
