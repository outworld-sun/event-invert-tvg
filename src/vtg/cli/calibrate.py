"""Freeze training-only matched-control quotas from two coverage runs."""

import argparse
import json

from vtg.preference.diagnostics import calibrate_matched
from vtg.utils.io import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", required=True, help="N1 coverage pair_diagnostics.jsonl")
    parser.add_argument("--event", required=True, help="N2 coverage pair_diagnostics.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-questions", required=True, type=int)
    args = parser.parse_args()
    report = calibrate_matched(
        read_jsonl(args.ordinary), read_jsonl(args.event), args.output, args.target_questions
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
