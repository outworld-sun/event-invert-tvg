"""Run one benchmark, optionally sharded with torchrun."""

import argparse

from vtg.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", help="Actor adapter directory: checkpoint-N or model")
    selection.add_argument(
        "--base", action="store_true", help="Explicitly evaluate the original base model"
    )
    parser.add_argument("--split", required=True, choices=["charades", "tvgbench"])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limit", type=int, help="Smoke-test subset, labeled partial_benchmark in metrics"
    )
    args = parser.parse_args()
    from vtg.evaluation.runner import run

    run(
        load_config(args.config, args.paths),
        args.checkpoint,
        args.split,
        args.output,
        args.base,
        args.limit,
    )


if __name__ == "__main__":
    main()
