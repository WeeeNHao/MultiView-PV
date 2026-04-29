from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

from pipeline import run_pipeline
from utils.config import load_runtime_config


def _parse_views(spec: str) -> List[str]:
    items: List[str] = []
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if start > end:
                start, end = end, start
            for v in range(start, end + 1):
                items.append(str(v))
            continue
        if part.lower() == "all":
            items.append("all")
            continue
        items.append(str(int(part)))

    seen = set()
    ordered: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered


def _build_run_opts(
    label: str,
    run_inference: bool,
    final_shp: str,
    collected_shp: str,
    multiview_shp: str,
    selected_shp: str,
    prompt_dir: str,
    prompt_txt: str,
) -> List[str]:
    opts = [
        f"pipeline.run_inference={'true' if run_inference else 'false'}",
        f"output.final_merged_shp={final_shp}",
        "output.trace_exports.enabled=false",
        f"output.trace_exports.collected_shp={collected_shp}",
        f"output.trace_exports.multiview_shp={multiview_shp}",
        f"output.trace_exports.selected_shp={selected_shp}",
        f"postprocess.prompt_export.output_dir={prompt_dir}",
        f"postprocess.prompt_export.output_txt={prompt_txt}",
    ]

    # Skip NCCL barrier risks in postprocess-only runs.
    if not run_inference:
        opts.append("distributed.enabled=false")

    if label == "all":
        opts.append("postprocess.view_selection.enabled=false")
    else:
        opts.append("postprocess.view_selection.enabled=false")
        opts.append(f"postprocess.view_selection.view_num={int(label)}")

    return opts


def _resolve_out_dir(config_path: str, out_dir: str | None) -> str:
    if out_dir:
        return out_dir

    cfg = load_runtime_config(config_path=config_path, opts=[]).raw
    output_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}
    final_merged = str(output_cfg.get("final_merged_shp", "outputs/final_merged.shp"))
    parent = os.path.dirname(final_merged) or "."
    return parent


def _make_paths(run_dir: str, label: str) -> Tuple[str, str, str, str, str, str]:
    os.makedirs(run_dir, exist_ok=True)
    final_shp = os.path.join(run_dir, f"view_{label}.shp")
    collected_shp = os.path.join(run_dir, f"view_{label}_collected.shp")
    multiview_shp = os.path.join(run_dir, f"view_{label}_multiview.shp")
    selected_shp = os.path.join(run_dir, f"view_{label}_selected.shp")
    prompt_dir = os.path.join(run_dir, "prompts")
    prompt_txt = os.path.join(run_dir, f"view_{label}_prompts.txt")
    return final_shp, collected_shp, multiview_shp, selected_shp, prompt_dir, prompt_txt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run postprocess view-number sweep without rerunning inference."
    )
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument(
        "--views",
        default="1-15,20,all",
        help="View sweep spec, e.g. '1-15,20,all'",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for sweep results. Default: <final_merged_dir>/view_sweep",
    )
    parser.add_argument(
        "--infer-first",
        type=int,
        default=0,
        choices=[0, 1],
        help="Whether to run inference in the first sweep run (1=yes, 0=no).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = _resolve_out_dir(config_path=args.config, out_dir=args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    labels = _parse_views(args.views)
    total = len(labels)
    if total == 0:
        raise ValueError("No valid views parsed from --views")

    print(f"[view-sweep] config: {args.config}")
    print(f"[view-sweep] out-dir: {out_dir}")
    print(f"[view-sweep] plans: {', '.join(labels)}")

    for idx, label in enumerate(labels, start=1):
        run_dir = os.path.join(out_dir, f"view_{label}")
        final_shp, collected_shp, multiview_shp, selected_shp, prompt_dir, prompt_txt = _make_paths(run_dir, label)
        run_inference = bool(args.infer_first) and idx == 1
        opts = _build_run_opts(
            label=label,
            run_inference=run_inference,
            final_shp=final_shp,
            collected_shp=collected_shp,
            multiview_shp=multiview_shp,
            selected_shp=selected_shp,
            prompt_dir=prompt_dir,
            prompt_txt=prompt_txt,
        )

        print(f"\n[view-sweep] ({idx}/{total}) running view={label}")
        print(f"[view-sweep] run_inference={run_inference}")
        print(f"[view-sweep] final: {final_shp}")
        runtime_cfg = load_runtime_config(config_path=args.config, opts=opts)
        run_pipeline(runtime_cfg)

    print("\n[view-sweep] done")


if __name__ == "__main__":
    main()