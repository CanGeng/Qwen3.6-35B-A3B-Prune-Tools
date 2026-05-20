"""HuggingFace snapshot download with optional mirror endpoint."""

from __future__ import annotations

import os
from pathlib import Path

# 取消顶部的导入
# from huggingface_hub import snapshot_download

def download_model(
        model_id: str,
        local_dir: str | Path,
        revision: str = "main",
        hf_endpoint: str | None = None,
) -> Path:
    local_dir = Path(local_dir).resolve()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    # 注入加速下载的环境变量（极大地提高 35B 这种大模型的下载速度）
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # 注入镜像环境变量
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")

    # 延迟导入
    from huggingface_hub import snapshot_download

    print(f"开始使用加速通道下载 {model_id}...")
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        max_workers=8,  # 增加并发数，进一步压榨带宽
        # resume_download=True, # 如果你用的 huggingface_hub 比较老可以加上这行，新版默认开启断点续传
    )
    return local_dir

