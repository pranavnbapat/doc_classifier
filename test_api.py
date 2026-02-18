#!/usr/bin/env python3
"""
Test script for Doc Classifier API v2.0 with Basic Auth

Tests health, subcategories, and classification with various fusion options.
"""

from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Import the app
from app import app

# Create test client
client = TestClient(app)

# Auth credentials (from .env file)
AUTH_USER = "nifty_chandrasekhar"
AUTH_PASSWORD = "3C11TCYVnqXJ"


def test_no_auth():
    """Test that endpoints reject requests without auth."""
    print("Testing without auth...")
    resp = client.get("/")
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code == 401:
        print("  ✓ Correctly rejected (401)")
        print(f"  WWW-Authenticate: {resp.headers.get('WWW-Authenticate')}")
        return True
    else:
        print(f"  ✗ Expected 401, got {resp.status_code}")
        return False


def test_wrong_auth():
    """Test that endpoints reject requests with wrong password."""
    print("Testing with wrong password...")
    resp = client.get("/", auth=(AUTH_USER, "wrongpassword"))
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code == 401:
        print("  ✓ Correctly rejected (401)")
        return True
    else:
        print(f"  ✗ Expected 401, got {resp.status_code}")
        return False


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    resp = client.get("/health", auth=(AUTH_USER, AUTH_PASSWORD))
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"  ✗ Failed: {resp.text}")
        return False
    
    data = resp.json()
    print(f"  Authenticated as: {data.get('authenticated_user')}")
    print(f"  Auth enabled: {data.get('auth_enabled')}")
    
    models = data.get('models', {})
    print(f"  Heuristics: {models.get('heuristics', {}).get('available')}")
    print(f"  Text LLM: {models.get('text_llm', {}).get('configured')}")
    print(f"  Vision LLM: {models.get('vision_llm', {}).get('configured')}")
    
    print("  ✓ Health check passed\n")
    return True


def test_subcategories():
    """Test subcategories endpoint."""
    print("Testing /subcategories endpoint...")
    resp = client.get("/subcategories", auth=(AUTH_USER, AUTH_PASSWORD))
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"  ✗ Failed: {resp.text}")
        return False
    
    data = resp.json()
    print(f"  Total subcategories: {data.get('total')}")
    print("  ✓ Subcategories endpoint passed\n")
    return True


def test_classify(pdf_path: str, options: dict, label: str):
    """Test classification with given options."""
    print(f"Testing /classify ({label}) with {Path(pdf_path).name}...")
    
    with open(pdf_path, 'rb') as f:
        resp = client.post(
            "/classify",
            files={"file": (Path(pdf_path).name, f, "application/pdf")},
            params=options,
            auth=(AUTH_USER, AUTH_PASSWORD)
        )
    
    print(f"  Status: {resp.status_code}")
    
    if resp.status_code != 200:
        print(f"  Error: {resp.text}")
        print("  ✗ Classification failed\n")
        return False
    
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


def main():
    """Run all tests."""
    print("=" * 70)
    print("Doc Classifier API v2.0 Test Suite")
    print("=" * 70)
    print(f"Auth User: {AUTH_USER}")
    print()
    
    # Auth tests
    if not test_no_auth():
        sys.exit(1)
    if not test_wrong_auth():
        sys.exit(1)
    if not test_health():
        sys.exit(1)
    if not test_subcategories():
        sys.exit(1)
    
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


if __name__ == "__main__":
    main()
