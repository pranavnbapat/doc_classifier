#!/usr/bin/env python3
"""
Test script for Doc Classifier API v2.0

Tests health, subcategories, and classification with various fusion options.
"""

import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    
    models = data.get('models', {})
    print(f"  Heuristics: {models.get('heuristics', {}).get('available')}")
    print(f"  Text LLM: {models.get('text_llm', {}).get('configured')}")
    print(f"  Vision LLM: {models.get('vision_llm', {}).get('configured')}")
    
    assert resp.status_code == 200
    print("  ✓ Health check passed\n")


def test_subcategories():
    """Test subcategories endpoint."""
    print("Testing /subcategories endpoint...")
    resp = requests.get(f"{BASE_URL}/subcategories")
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Total subcategories: {data.get('total')}")
    assert resp.status_code == 200
    print("  ✓ Subcategories endpoint passed\n")


def test_classify(pdf_path: str, options: dict, label: str):
    """Test classification with given options."""
    print(f"Testing /classify ({label}) with {Path(pdf_path).name}...")
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
        resp = requests.post(f"{BASE_URL}/classify", files=files, params=options)
    
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        
        # Best match
        best = data.get('best_match', {})
        print(f"\n  📊 BEST MATCH:")
        print(f"     {best.get('subcategory_name')} (conf: {best.get('confidence'):.3f})")
        
        # Fusion info
        fusion = data.get('fusion')
        if fusion and fusion.get('fused'):
            print(f"\n  🔗 FUSION:")
            print(f"     Strategy: {fusion.get('strategy')}")
            print(f"     Weights: {fusion.get('weights')}")
            print(f"     Agreement: {fusion.get('agreement_score'):.2f}")
        
        # Individual sources
        if data.get('heuristics'):
            h = data['heuristics']
            print(f"\n  🧮 HEURISTICS:")
            print(f"     {h.get('subcategory_name')} (conf: {h.get('confidence'):.3f})")
        
        if data.get('vision_llm') and 'error' not in data.get('vision_llm', {}):
            v = data['vision_llm']
            print(f"\n  👁️  VISION LLM:")
            print(f"     {v.get('subcategory_name')} (conf: {v.get('confidence'):.3f})")
        
        if data.get('text_llm') and 'error' not in data.get('text_llm', {}):
            t = data['text_llm']
            print(f"\n  📝 TEXT LLM:")
            print(f"     {t.get('subcategory_name')} (conf: {t.get('confidence'):.3f})")
        
        proc = data.get('processing_info', {})
        print(f"\n  ⏱️  Processing time: {proc.get('processing_time_ms', 0):.1f}ms")
        print("  ✓ Classification passed\n")
        return True
    else:
        print(f"  Error: {resp.text}")
        print("  ✗ Classification failed\n")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Doc Classifier API v2.0 Test Suite")
    print("=" * 70)
    print(f"Base URL: {BASE_URL}\n")
    
    try:
        test_health()
        test_subcategories()
        
        # Find sample PDFs
        sample_dir = Path("files")
        pdf_files = list(sample_dir.glob("*.pdf")) if sample_dir.exists() else []
        
        if not pdf_files:
            print("No sample PDFs found in 'files/' directory")
            sys.exit(1)
        
        pdf = str(pdf_files[0])
        
        # Test cases
        test_classify(pdf, {}, "heuristics only")
        test_classify(pdf, {"use_text_llm": "true", "heuristics_alpha": "0.4"}, "heuristics + text (alpha=0.4)")
        test_classify(pdf, {"use_vision": "true", "vision_max_pages": "5"}, "heuristics + vision")
        test_classify(pdf, {"use_vision": "true", "use_text_llm": "true", "fusion_strategy": "adaptive"}, "all sources + adaptive fusion")
        
        print("=" * 70)
        print("All tests completed!")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to {BASE_URL}")
        print("  Make sure the server is running:")
        print("    python start_server.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
