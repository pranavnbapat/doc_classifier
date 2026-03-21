# Document Subcategories Consolidation Plan

## Scope

This file focuses only on document-like material and treats the metadata guide PDF as canonical.

It is aligned with [category_auto_selection_policy.md](./category_auto_selection_policy.md):

- `Document` is an operational category for `file-based KOs`
- `Slideshow/Presentation` is removed as a standalone category
- slide decks are treated as `Document > Presentation`

Important source note:
- the guide includes `Problem-solving presentation`
- `data_model.subcategories.json` does not

The consolidation below absorbs that missing guide-defined subtype into `Presentation`.

---

## Decision

### Remove standalone category

- `Slideshow/Presentation`

### Fold into

- `Document`

### Why

For later category auto-selection, `Slideshow/Presentation` is not operationally strong enough as a top-level category:

- slide decks are often uploaded as the same file types as documents,
- its current subtypes are mostly purpose labels rather than stable content forms,
- and a single deck may be informative, educational, motivational, and decision-supporting at the same time.

So the cleaner model is:

- keep `Presentation` as a **document subcategory**
- do not keep `Slideshow/Presentation` as a separate top-level category

This keeps the document taxonomy consistent with the category policy and with later document-classification logic.

---

## Final Consolidated Document Taxonomy

### Keep Separate

1. `Article in conference proceedings`
2. `Journal article`
3. `Chapter in edited volume`
4. `Book`
5. `Thesis`
6. `Tutorial`

### Consolidate

1. `Booklet` + `Brochure` + `Factsheet` + `Flyer`
   -> `Informational Booklet`

2. `Handbook` + `Manual` + `Guide`
   -> `Guide/Manual`

3. `Deliverable report` + `Milestone report` + `Report/paper` + `Review document` + `Technical/technology article` + `Technical information/specifications card`
   -> `Technical Report`

4. `Practice abstract` + `Decision-making presentation` + `Educational/training presentation` + `Informative presentation` + `Motivational presentation` + `Problem-solving presentation`
   -> `Presentation`

5. `Newsletter` + `Press release` + `Policy brief`
   -> `News & Communication`

---

## Explicit Mapping

This section states explicitly which source subcategories were consolidated into which target subcategory, and which source subcategories were deliberately kept unchanged.

### Consolidated into `Informational Booklet`

- `Booklet`
- `Brochure`
- `Factsheet`
- `Flyer`

Why:
- all four are short-form information carriers,
- the distinctions are mainly page-count, layout, or promotional nuance,
- and those distinctions are too weak to justify separate classifier targets.

### Consolidated into `Guide/Manual`

- `Handbook`
- `Manual`
- `Guide`

Why:
- these are near-synonyms in document form,
- they all function as reference-oriented instructional material,
- and the split is more terminological than semantic.

### Consolidated into `Technical Report`

- `Deliverable report`
- `Milestone report`
- `Report/paper`
- `Review document`
- `Technical/technology article`
- `Technical information/specifications card`

Why:
- these labels mostly vary by project workflow, packaging, or length,
- not by a strong difference in retrieval intent,
- and they can be grouped under one formal technical/project-document family.

### Consolidated into `Presentation`

- `Practice abstract`
- `Decision-making presentation`
- `Educational/training presentation`
- `Informative presentation`
- `Motivational presentation`
- `Problem-solving presentation`

Why:
- the stable commonality is slide-based presentation form,
- the original distinctions are purpose labels,
- and one presentation can satisfy several of those purposes simultaneously.

### Consolidated into `News & Communication`

- `Newsletter`
- `Press release`
- `Policy brief`

Why:
- all three are concise communication artifacts,
- they differ more by audience and publishing context than by content form,
- and they are better grouped together than separated into small classifier buckets.

### Kept unchanged

- `Article in conference proceedings`
- `Journal article`
- `Chapter in edited volume`
- `Book`
- `Thesis`
- `Tutorial`

Why these were kept:

- `Article in conference proceedings`, `Journal article`, `Chapter in edited volume`, `Book`, and `Thesis` are all academically meaningful distinctions with different structural and bibliographic signals.
- `Tutorial` was kept separate because it has stronger procedural and step-by-step cues than `Guide/Manual`, making it both user-meaningful and classifier-friendly.

---

## Rationale By Group

### Informational Booklet

Merged:
- `Booklet`
- `Brochure`
- `Factsheet`
- `Flyer`

Rationale:
- all four are short-form information carriers,
- the distinction is mostly page count, layout, or promotional tone,
- and users typically do not benefit from choosing between these micro-variants.

Measurable classification direction:
- short page count
- brochure/flyer-like layout
- concise promotional or summary language
- low academic or formal-report structure

### Guide/Manual

Merged:
- `Handbook`
- `Manual`
- `Guide`

Rationale:
- these are near-synonymous in document form,
- they all act as reference-oriented instructional documents,
- and separating them creates arbitrary uploader choices.

Measurable classification direction:
- instructional/reference wording
- sectioned guidance structure
- materials, procedures, cautions, or reference sections
- lower emphasis on explicit training progression than `Tutorial`

### Technical Report

Merged:
- `Deliverable report`
- `Milestone report`
- `Report/paper`
- `Review document`
- `Technical/technology article`
- `Technical information/specifications card`

Rationale:
- these labels mostly reflect project workflow, length, or packaging,
- not materially different retrieval behaviour,
- and they all belong to one family of formal technical/project documentation.

Measurable classification direction:
- project and deliverable markers
- formal section structure
- executive summary / appendix / revision cues
- technical terminology and specification-like language
- evidence-backed or review-oriented prose

### Presentation

Merged:
- `Practice abstract`
- `Decision-making presentation`
- `Educational/training presentation`
- `Informative presentation`
- `Motivational presentation`
- `Problem-solving presentation`

Rationale:
- the stable commonality is slide-based presentation format,
- while the existing distinctions are purpose labels,
- and one deck can realistically satisfy several of those purposes at once.

Measurable classification direction:
- slide-like page structure
- short text blocks / bullet density
- title-heavy pages
- visual-heavy layout
- page-by-page presentation rhythm rather than continuous prose

### News & Communication

Merged:
- `Newsletter`
- `Press release`
- `Policy brief`

Rationale:
- all three are concise communication artifacts,
- they differ more by audience and publishing context than by content form,
- and they are better retrieved together than split apart.

Measurable classification direction:
- concise update/announcement framing
- release or communication language
- low procedural depth
- shorter format than technical reports or books

---

## Final Count

- raw `Document` subcategories in current JSON: `23`
- guide-defined slideshow subcategories: `7`
- document-like source total for consolidation: `30`
- final consolidated `Document` subcategories: `11`

Final list:

1. `Article in conference proceedings`
2. `Journal article`
3. `Chapter in edited volume`
4. `Book`
5. `Thesis`
6. `Informational Booklet`
7. `Guide/Manual`
8. `Tutorial`
9. `Technical Report`
10. `Presentation`
11. `News & Communication`

---

## Measurable Criteria Matrix

This matrix is meant as the transition point between taxonomy consolidation and reproducible classification design.

Operational note:
- the code now carries the same criteria hints inside [subcategories.py](/home/pranav/PyCharm/EU-FarmBook/ko_classifier/docint/rubrics/subcategories.py)
- the matrix below should therefore be treated as the human-readable view of the same intended classification logic

| Consolidated Subcategory | Strong Positive Signals | Helpful Secondary Signals | Typical Negative / Competing Signals |
|--------------------------|-------------------------|---------------------------|--------------------------------------|
| `Article in conference proceedings` | conference/proceedings markers, citation density, research structure | references section, author/venue cues | no conference cues, strongly booklet/news-like format |
| `Journal article` | IMRaD structure, citation density, abstract quality, peer-review markers | references heading, DOI cues | short promotional format, slide layout, no academic structure |
| `Chapter in edited volume` | book features, citations, chapter/book cues | edited-by / publisher language | conference/journal cues stronger than book cues |
| `Book` | ISBN/publisher/chapter cues, long form, formal structure | table of contents, chapter progression | short-form communication or presentation layout |
| `Thesis` | thesis markers, university/degree cues, formal structure, citations | acknowledgements, references, dissertation language | very short form, no academic markers |
| `Informational Booklet` | short form, concise informational/promotional wording, visually segmented layout | brochure/flyer style, low formal structure | strong policy/news signals, clear slide markers, deep technical/report structure |
| `Guide/Manual` | guidance/reference wording, procedure/materials/safety cues, structured sections | checklist cues, formal instructional headings | explicit tutorial progression stronger than reference tone, presentation-like layout |
| `Tutorial` | explicit steps, how-to framing, learning progression, examples/exercises | learning objectives, task completion language | reference/manual tone without progression, strong presentation layout |
| `Technical Report` | deliverable/project markers, technical terminology, formal structure, version/revision cues | executive summary, appendix, review/specification language | short promotional/news style, strong academic publication cues |
| `Presentation` | slide indicators, visual-heavy layout, title-heavy pages, short text blocks | short form, page rhythm of slides | strong policy/news cues, no slide markers, continuous prose structure |
| `News & Communication` | timeliness/update framing, release/byline format, regulatory/policy update cues | governance/compliance references, short form | strong slide cues, long formal report structure, academic citation structure |

### How to use the matrix

For each candidate subcategory, the classifier should ideally evaluate:

1. positive evidence present,
2. missing expected evidence,
3. competing evidence that fits a neighboring class better.

This is the basis for both:

- a defensible classification decision, and
- a contrastive explanation of why the winner beat the nearest alternatives.

---

## Why Tutorial Stays Separate

`Tutorial` was not merged into `Guide/Manual` because it carries a clearer step-by-step learning intent.

That distinction is meaningful both for users and for later classification logic:

- `Guide/Manual` is reference-oriented
- `Tutorial` is procedural and instructional

That difference is worth preserving.

Measurable classification direction for `Tutorial`:

- explicit steps or ordered progression
- “how to” or lesson-style framing
- task completion focus
- examples or exercises

---

## Assessment Of `Document_type_list_review_suggestion.md`

The colleague suggestion is directionally useful but should not be followed exactly.

### What is useful

- `Project report` and `Technical report` were separated instead of being left as many tiny labels
- the redundancy among short printed materials was correctly noticed
- the need for broader instructional groupings was correctly identified

### What should not be adopted directly

- `Information/promotional material` groups booklet/flyer-style artifacts with newsletter/press release, but these have different measurable cues
- `Instructional material` merges `tutorial` with `guide/manual`, which weakens later classification clarity
- `Scientific publication/grey literature` is too broad for a system that wants measurable, reproducible document classification
- `Education/training material` and `Template` introduce new vocabulary that is not grounded in the metadata guide and not yet justified by the current source set

### Final position

The colleague note is useful as a signal that consolidation is necessary.
The current document proposal is more defensible because it:

- stays closer to the metadata guide,
- aligns with the category policy,
- and preserves distinctions that are more likely to support measurable classification later.
