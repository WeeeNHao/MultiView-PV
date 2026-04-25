from __future__ import annotations

from refactor_v2.config import load_runtime_config, parse_args
from refactor_v2.pipeline import run_pipeline


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(config_path=args.config, opts=args.opts)
    run_pipeline(runtime_cfg)


if __name__ == "__main__":
    main()
