from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from run_experiment import run


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the DDQN-PER talent-project matching experiments."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="Path to a local CSV or directory containing the CSV.")
    parser.add_argument("--download", action="store_true", help="Download the configured Kaggle dataset with kagglehub.")
    parser.add_argument("--smoke", action="store_true", help="Fast execution test only; not for manuscript reporting.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.output:
        out_dir = Path(args.output)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(config["experiment"]["output_root"]) / f"{config['experiment']['name']}_{stamp}"

    completed = run(
        config=config,
        data_path=args.data,
        download=args.download,
        output_dir=out_dir,
        smoke=args.smoke,
    )

    print(json.dumps({
        "status": "completed",
        "output_directory": str(completed),
        "smoke_mode": bool(args.smoke),
    }, indent=2))


if __name__ == "__main__":
    main()
