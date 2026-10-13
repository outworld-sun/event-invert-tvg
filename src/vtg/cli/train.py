"""Explicit training flags and optional JSON defaults share one validated entry."""

import json
import os

from vtg.cli.arguments import resolve_arguments, training_parser


def main(argv=None):
    parser = training_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_arguments(args, parser)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.print_config:
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    from vtg.training.runner import preflight, run

    if args.dry_run:
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        return
    if not args.output and not args.prepare_only:
        parser.error("--output_dir is required for training")
    if args.prepare_data == "auto" or args.prepare_only:
        from vtg.data.lifecycle import ensure_training_data

        ensure_training_data(config)
    if args.prepare_only:
        return
    run(config, args.output, args.resume, args.stop_after, args.coverage_size)


if __name__ == "__main__":
    main()
