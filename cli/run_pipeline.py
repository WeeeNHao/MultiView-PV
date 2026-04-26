from __future__ import annotations

from utils.config import load_runtime_config, parse_args
from pipeline import run_pipeline


def main() -> None:
    args = parse_args()
    runtime_cfg = load_runtime_config(config_path=args.config, opts=args.opts)
    run_pipeline(runtime_cfg)


if __name__ == "__main__":
    main()
