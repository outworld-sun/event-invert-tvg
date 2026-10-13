"""Prepare training manifests and independent benchmark inputs."""

import argparse
import json

from vtg.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths")
    parser.add_argument("--benchmarks-only", action="store_true")
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    if args.benchmarks_only and args.training_only:
        parser.error("Choose at most one preparation scope")
    config = load_config(args.config, args.paths)
    if not args.benchmarks_only:
        from vtg.data.prepare import prepare

        print(json.dumps(prepare(config), ensure_ascii=False, indent=2))
    if not args.training_only:
        from vtg.data.benchmarks import prepare_benchmarks

        print(json.dumps(prepare_benchmarks(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
