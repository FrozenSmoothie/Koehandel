"""
Debug script to see what metrics are actually available
"""

import json
from pathlib import Path


def check_metrics():
    """Check what metrics are in the result file."""
    result_file = Path("./results_test/quick_test/PPO_koehandel_env_*/result.json")

    # Find the actual file
    results = list(Path("./results_test/quick_test").glob("PPO_koehandel_env_*/result.json"))

    if not results:
        print("No result files found")
        return

    result_file = results[0]
    print(f"Reading: {result_file}")
    print("=" * 70)

    with open(result_file, 'r') as f:
        lines = f.readlines()
        if lines:
            # Get last line
            metrics = json.loads(lines[-1])

            print("\n[ALL AVAILABLE METRICS]")
            print("=" * 70)

            # Show all metric keys
            all_keys = sorted(metrics.keys())

            # Filter for relevant ones
            relevant_keys = []
            for key in all_keys:
                if any(x in key.lower() for x in ['step', 'episode', 'reward', 'loss', 'sample', 'train']):
                    relevant_keys.append(key)

            print("\n[RELEVANT METRICS]")
            for key in relevant_keys:
                value = metrics.get(key, "N/A")
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")

            print("\n[STEP-RELATED METRICS]")
            for key in all_keys:
                if 'step' in key.lower():
                    value = metrics.get(key, "N/A")
                    print(f"  {key}: {value}")

            print("\n[SAMPLE-RELATED METRICS]")
            for key in all_keys:
                if 'sample' in key.lower():
                    value = metrics.get(key, "N/A")
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    check_metrics()