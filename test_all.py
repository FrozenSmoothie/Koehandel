"""
Run all tests and generate a report
"""

import sys
import subprocess

def run_test(test_file, description):
    """Run a test file and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"[PASS] {description}")
            return True
        else:
            print(f"[FAIL] {description}")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description}")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("KOEHANDEL RL ENVIRONMENT - TEST SUITE")
    print("="*60)
    print(f"Date: 2025-10-21")
    print(f"Developer: FrozenSmoothie")

    tests = [
        ("test_basic.py", "Basic Functionality Test"),
        ("test_api_compliance.py", "PettingZoo API Compliance Test"),
        ("test_suite.py", "Unit Tests Suite"),
    ]

    results = []
    for test_file, description in tests:
        passed = run_test(test_file, description)
        results.append((description, passed))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for description, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {description}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n[SUCCESS] All tests passed! Environment is ready for training.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())