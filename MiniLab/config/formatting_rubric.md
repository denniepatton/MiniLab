# Formatting & Reporting Guidelines (Nature-style)

## Applicability
- Apply **globally**.
- If a required section is not present (e.g., “Methods” in a lit review), ignore section-structure rules and keep **style + citations + numeric reporting + figure integrity** rules.

## Global document defaults (any deliverable)
- MUST: use a **Helvetica 11 pt** font for narrative text.
- MUST: **double-space** body text.
- MUST: include **line numbers** when producing a review-ready document.
- MUST: use **consistent heading hierarchy** (no skipped levels).

## Writing & readability (any deliverable)
- MUST: write for a **broad scientific audience**; minimize jargon; define essential terms at first use.
- MUST: minimize nonstandard abbreviations; define at first use; avoid acronym overload.
- SHOULD: prefer **clear, short sentences**; use active voice when it improves clarity.
- MUST: avoid overstated causal claims unless supported by study design and/or evidence.

## Headings & structure (when sectioned)
- MUST: use a stable hierarchy: `H1` (document title) → `H2` (major sections) → `H3` (subsections).
- SHOULD: keep subheadings short and descriptive, approximately one line of text.
- MAY: omit formal headings in a lit review if a narrative flow is preferred; still keep logical paragraphing.

## References & citation mechanics (any deliverable with citations)
- MUST: use **numbered references** (Vancouver-like).
- MUST: references are **sequential** in first-appearance order across:
  - main text
  - methods
  - tables
  - figures/extended data legends
- MUST: in-text citations are **superscript numbers** (e.g., `...as shown previously.^3`).
- MUST (reference list formatting):
  - include **article titles**
  - one publication per reference number
  - list **all authors unless >5**, then first author + `et al.`
- MUST: include a **clickable** doi link for each reference.
- SHOULD: avoid auto-generated “field codes” if exporting to Word; provide plain text references.

## Numbers, units, and general numeric hygiene (any deliverable)
- MUST: use **SI units** or field-standard units; be consistent.
- MUST: use a **space** between number and unit (e.g., `10 mm`, `5 µg`).
- MUST: use commas for thousands (e.g., `1,000`).
- SHOULD: report summary statistics with clear semantics (e.g., mean ± s.d., median [IQR]).
- MUST: define all symbols and abbreviations used in tables/figures.

## Statistical reporting (when quantitative claims are made)
- MUST: report **exact n** and what n represents (biological replicates vs technical replicates vs repeated measures).
- MUST: state whether tests are **one- or two-sided**.
- MUST: report **exact P values** where appropriate (not only thresholds like `P < 0.05`).
- SHOULD: include effect sizes and uncertainty (e.g., **CI**) alongside P values.
- MUST: specify:
  - statistical test used
  - any covariates/model terms
  - multiple-testing correction (if applicable)
  - assumptions checks or robustness strategy (if relevant)
- MUST (figures/legends): define error bars (s.d., s.e.m., CI) and what they summarize.

## Methods content requirements (Methods-only or any document with Methods)
- MUST: include enough detail to enable interpretation and replication.
- SHOULD: subdivide with short, bold-like subsection headings (e.g., Sample prep, Sequencing, Preprocessing, Modeling, Statistics).
- MUST: clearly specify:
  - materials/reagents (source, identifiers when relevant)
  - software + versions
  - parameters and thresholds (not “default” without stating what default is)
  - data inclusion/exclusion criteria
  - randomization/blinding (if applicable)
- SHOULD: keep Methods free of figures/tables; place supporting material in an appendix/extended section if needed.
- SHOULD: include **Data availability** and **Code availability** statements when the deliverable implies publishable results.

## Figures: layout, typography, export (Figures-only or any document with figures)
### Layout & sizing
- MUST: keep figures **simple and legible**; remove non-essential decoration.
- SHOULD: design for reduction to journal column widths:
  - single-column target: ~90 mm
  - double-column target: ~180 mm
- MUST: multi-panel figures must use consistent scaling and alignment.
- MUST: use **scale bars** (not magnification factors) where applicable.

### Color-blind-friendly plot palette (HEX; ordered for incremental use)
- Purpose: categorical series colors that remain distinguishable under common color-vision deficiencies.
- Ordering: pick the **top N** for N=2,3,5,7… and they will remain reasonably complementary.
#### Primary sequence (recommended)
1. Blue:   `#0072B2`
2. Orange: `#E69F00`
3. Green:  `#009E73`
4. Purple: `#CC79A7`
5. Sky:    `#56B4E9`
6. Vermilion / Red-orange: `#D55E00`
7. Yellow: `#F0E442`
8. Gray:   `#7F7F7F`  (use for baselines/controls, not a “main” category if avoidable)
#### Neutral + emphasis helpers (optional)
- Dark gray (text/axes): `#4D4D4D`
- Light gray (gridlines): `#D9D9D9`
- Near-black: `#1A1A1A`
- White: `#FFFFFF`
#### Usage notes (keep simple)
- Prefer ≤7 categorical colors per plot; beyond that, add shape/linetype encoding.
- Avoid Yellow (`#F0E442`) for thin lines on white; reserve for fills or thick lines.
- Use "happy" and "sad" colors appropriately (e.g. Blue for "responders" and Orange for "non-responders")
- Be consistent with colors used for labeling; if Green is used for "female" in Figure 1A, it should be used for "female" in all other panels as well.

### Figure typography (hard defaults)
- MUST: use a single sans-serif font in figures (default: **Helvetica**; **Symbol** for Greek).
- SHOULD: target sizes at final print scale:
  - axes/labels: **5–7 pt**
  - panel labels (A, B, …): **~8 pt bold**
- MUST: keep text readable without zooming; avoid text over complex backgrounds.

### Export specs (final-quality)
- SHOULD: prefer **vector** (PDF/SVG/EPS) with editable layers for plots/diagrams.
- MUST: raster components at **≥300 dpi** at final size.
- SHOULD: use **RGB** color mode unless a specific workflow requires otherwise.

## Figure legends (any deliverable with figures)
- MUST: each legend begins with a short **title/descriptor**, then explains panels and symbols.
- MUST: define:
  - what each panel shows
  - symbols/colors/lines
  - statistical details and error bars (if any)
  - n, tests, and corrections when relevant
- MUST: legends do **not** contain extended Methods; only essential context.
- SHOULD: keep legends concise (target <300 words per figure).
- Placement:
  - review-style compilation: legends may appear adjacent to figures
  - manuscript-style: legends can be grouped after references

## Tables (when used)
- MUST: provide a short title and define abbreviations/symbols in footnotes.
- SHOULD: keep orientation portrait; avoid rotated tables.
- MUST: ensure numbers are aligned and units are explicit.

## Extended/Appendix material (optional)
- SHOULD: use an “Extended Data” / “Appendix” section for:
  - supplementary validations
  - extra controls
  - additional cohort breakdowns
  - parameter sweeps
- MUST: keep formatting consistent with main figures/tables.

## Image integrity (non-negotiable when using image data)
- MUST: images must represent original data faithfully; retain originals.
- MUST NOT: use cloning/healing/touch-up tools to alter content.
- MUST: disclose non-linear adjustments and pseudocoloring when used.
- MUST: if combining images from different fields/times, **demarcate boundaries** and state it in the legend.

## Minimal compliance profile (for short outputs like lit reviews)
- MUST: clear writing + defined terms
- MUST: sequential numbered citations + clean reference list
- MUST: consistent units and numeric reporting
- MUST: figure typography + integrity (if figures included)
- SHOULD: light heading hierarchy for readability
