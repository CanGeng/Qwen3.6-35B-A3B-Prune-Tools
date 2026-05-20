"""Step 0: download teacher model from HuggingFace (optional mirror)."""

from __future__ import annotations

import argparse
from pathlib import Path

from moe_prune_distill.config import load_config
from moe_prune_distill.utils.download import download_model
from moe_prune_distill.utils.logging import get_logger


def validate_downloaded(local_dir: Path) -> None:
    cfg = local_dir / "config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"Missing config.json under {local_dir}")
    st = list(local_dir.glob("*.safetensors"))
    if not st:
        raise FileNotFoundError(f"No *.safetensors under {local_dir}")


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Download HF MoE model")
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.download.local_dir).resolve()
    log.info("Downloading %s -> %s", cfg.download.model_id, out)
    download_model(
        cfg.download.model_id,
        out,
        revision=cfg.download.revision,
        hf_endpoint=cfg.download.hf_endpoint,
    )
    validate_downloaded(out)
    log.info("Download OK: %s", out)


if __name__ == "__main__":
    main()
