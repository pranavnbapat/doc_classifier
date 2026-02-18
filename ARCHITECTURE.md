# Doc Classifier Architecture

## Current Flow (Working Now)

```
PDF Upload
    │
    ├──► PyMuPDF extracts text (fast)
    │    └── If text quality poor → Tesseract OCR (slower)
    │
    ▼
Extracted Text (string)
    │
    ├──► Evidence-Based Classification (HEURISTICS)
    │    ├── Extract features (24+ signals)
    │    ├── Score each subcategory
    │    └── Return ALL 11 subcategories with probabilities
    │
    └──► [Optional] LLM Classification (Qwen)
         ├── Send extracted TEXT to LLM
         ├── Ask for same 11 subcategories
         └── Return LLM's ranking
```

## What We Have

### 1. Text Extraction (Working)
- **Primary**: PyMuPDF (fast, preserves structure)
- **Fallback**: Tesseract OCR (when PDF is scanned images)
- **Output**: Plain text + lines array

### 2. Evidence-Based Classification (Working)
- 24+ measurable features
- 11 subcategories from data_model
- Returns: ALL candidates with confidence + probability

### 3. LLM Integration (Partially Working)
- Currently: Sends extracted TEXT to Qwen
- Asks for classification into 11 subcategories
- Returns: LLM's best match + rationale

## The Problem You Identified

LLM was classifying into **OLD parent categories** (6 types) instead of **subcategories** (11 types). This is now **fixed** - both use the same 11 subcategories.

## Two LLM Options (Clarified)

You mentioned 2 LLMs on 2 servers. Here are the options:

### Option A: Text-Based (Qwen) - RECOMMENDED
**What happens:**
1. Extract text from PDF (PyMuPDF/OCR)
2. Run evidence-based classification
3. Send **same extracted text** to Qwen LLM
4. Ask Qwen to classify into same 11 subcategories
5. Compare results

**When to use:**
- Text-heavy documents
- Long documents (many pages)
- Speed is important
- Content matters more than layout

**API:**
```bash
curl -X POST "http://localhost:8000/classify?use_llm=true" \
  -F "file=@document.pdf"
```

### Option B: Vision-Based (InternVL) - EXPERIMENTAL
**What happens:**
1. Convert PDF pages to images
2. Send **images** directly to InternVL vision model
3. Vision model "sees" the document layout
4. Returns classification based on visual appearance

**When to use:**
- Documents with complex layouts
- Lots of images/diagrams
- Format/layout is important (presentations, posters)
- Text extraction fails (weird PDFs)

**API:**
```bash
curl -X POST "http://localhost:8000/classify?use_llm=true&llm_type=vision" \
  -F "file=@document.pdf"
```

## Recommendation

**Use Option A (Text-based)** for now because:
1. Simpler architecture
2. Both classifiers use same input (extracted text)
3. Easier to compare results
4. Faster processing
5. Works with your current Qwen setup

**Vision model** is a separate feature that can be added later if needed for specific document types.

## Simplified Architecture (Proposed)

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF UPLOAD                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TEXT EXTRACTION (PyMuPDF + OCR)                │
│              Extracts: text, lines, page count              │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│  EVIDENCE-BASED       │       │  LLM (Qwen)           │
│  CLASSIFICATION       │       │  (Optional)           │
│                       │       │                       │
│  • 24+ features       │       │  • Receives text      │
│  • 11 subcategories   │       │  • Same 11 categories │
│  • Probabilities      │       │  • Rationale          │
│  • Excerpts as proof  │       │                       │
└───────────────────────┘       └───────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              COMBINED RESPONSE                               │
│  • Best match (evidence-based)                              │
│  • All 11 subcategories with scores                         │
│  • LLM opinion (if requested)                               │
│  • Rationales from both                                     │
└─────────────────────────────────────────────────────────────┘
```

## Questions for You

1. **Do you want to use LLM at all?**
   - Yes → Keep current implementation
   - No → Remove LLM code, keep only evidence-based

2. **Do you need the vision model (InternVL)?**
   - Yes → Keep vision option
   - No → Remove vision code, simpler is better

3. **How should we combine results?**
   - Option 1: Show both separately (current)
   - Option 2: Weighted fusion (e.g., 60% evidence + 40% LLM)
   - Option 3: LLM only when evidence is weak

4. **What are your 2 LLMs?**
   - Server 1: Qwen (text)
   - Server 2: ??? (another Qwen? or InternVL?)
