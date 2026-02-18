# run_classify.py

from __future__ import annotations

import argparse
import json

from dataclasses import asdict

from docint.pipeline.classify import classify_pdf


def print_verdict(res) -> None:
    """
    Human-readable verdict from the machine-readable output.
    Keeps it short enough to skim in a terminal.
    """
    decision = res.reasons.get("decision", {})
    evidence_score = res.reasons.get("evidence_score")
    used_override = decision.get("used_llm_override", False)
    components = decision.get("evidence_components", {})

    probs = res.probs
    top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top1, top2, top3 = top[:3]

    print("\n" + "=" * 60)
    print(f"FINAL VERDICT: {res.label}  (confidence={res.confidence:.3f})")
    print(f"Top-3: {top1[0]}={top1[1]:.3f}, {top2[0]}={top2[1]:.3f}, {top3[0]}={top3[1]:.3f}")
    print(f"Decision mode: {'LLM override' if used_override else 'Fusion/heuristics'}")
    print(f"Evidence score: {evidence_score} | components: {components}")

    # Heuristics summary
    heur = res.reasons.get("heuristic_probs", {})
    heur_top = sorted(heur.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if heur_top:
        heur_str = ", ".join([f"{k}={v:.3f}" for k, v in heur_top])
        print(f"Heuristics (top-3): {heur_str}")

    # LLM summary
    llm = res.reasons.get("llm", {})
    if llm:
        llm_probs = llm.get("probs", {})
        llm_top = sorted(llm_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        llm_str = ", ".join([f"{k}={v:.3f}" for k, v in llm_top])
        print(f"LLM (top-3): {llm_str}")
        rationale = (llm.get("rationale") or "").strip()
        if rationale:
            print("\nWhy this label:")
            # Keep it readable in terminal
            print(f"- {rationale}")

    # Subcategory classification (NEW)
    subcat_data = res.reasons.get("subcategory_classification", {})
    best_subcat = subcat_data.get("best_match")
    if best_subcat:
        print(f"\n📄 SUBCATEGORY: {best_subcat['subcategory_name']} (confidence={best_subcat['confidence']:.3f})")
        print(f"   ID: {best_subcat['subcategory_id']}")
        print(f"   Features detected: {', '.join(best_subcat['features_found'])}")
        print(f"   Rationale: {best_subcat['rationale']}")
        
        # Show other possible subcategories
        all_scores = subcat_data.get("all_scores", [])
        if len(all_scores) > 1:
            print(f"   Other candidates:")
            for alt in all_scores[1:3]:  # Show next 2 alternatives
                print(f"     - {alt['subcategory_name']}: {alt['confidence']:.3f} ({', '.join(alt['features_found'][:2])})")
    else:
        print("\n📄 SUBCATEGORY: Could not determine (insufficient evidence)")

    # Optional: surface the main "missing evidence" signals
    sections = res.reasons.get("features", {}).get("sections", {}).get("counts", {})
    cites = res.reasons.get("features", {}).get("citations", {})
    if sections and cites:
        print("\nKey measurable cues:")
        print(f"- Sections detected: {sum(sections.values())} (IMRaD headings mostly absent)")
        print(f"- Citation markers: numeric={cites.get('cite_numeric', 0)}, authoryear={cites.get('cite_authoryear', 0)}, doi={cites.get('doi_mentions', 0)}")
        print(f"- URLs found: {cites.get('url_mentions', 0)}")

    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a PDF into document types with heuristics + optional LLM.")
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use heuristics only")
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract language(s), e.g. eng or eng+nld")
    parser.add_argument("--ocr-max-pages", type=int, default=10, help="Max pages to OCR when needed")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL (vLLM)")
    parser.add_argument("--llm-model", default="qwen3-30b-a3b-awq", help="Model name as exposed by the server")
    parser.add_argument("--llm-ensemble-n", type=int, default=5, help="Number of LLM runs to average")
    parser.add_argument("--fusion-alpha", type=float, default=0.6, help="LLM weight in fusion (0..1)")
    parser.add_argument("--no-json", action="store_true", help="Do not print full JSON output")
    parser.add_argument("--compact-json", action="store_true", help="Print only label/confidence/probs + decision info")

    args = parser.parse_args()

    res = classify_pdf(
        args.pdf_path,
        llm_enabled=not args.no_llm,
        ocr_lang=args.ocr_lang,
        ocr_max_pages=args.ocr_max_pages,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_ensemble_n=args.llm_ensemble_n,
        fusion_alpha=args.fusion_alpha,
    )

    print_verdict(res)

    if not args.no_json:
        if args.compact_json:
            compact = {
                "label": res.label,
                "confidence": res.confidence,
                "probs": res.probs,
                "decision": res.reasons.get("decision", {}),
                "final_decision": res.reasons.get("final_decision", {}),
            }
            print(json.dumps(compact, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(asdict(res), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
