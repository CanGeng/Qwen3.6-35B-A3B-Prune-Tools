"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ProjectConfig:
    name: str
    output_dir: str


@dataclass
class DownloadConfig:
    model_id: str
    revision: str
    hf_endpoint: str | None
    local_dir: str


@dataclass
class PruneConfig:
    target_num_experts: int
    target_num_experts_per_tok: int
    keep_shared_experts: bool
    expert_selection: Literal["first_n", "router_top", "manual"]
    manual_experts: dict[str, list[int]] | list[list[int]] | None
    router_stat_samples: int
    student_dir: str
    router_stats_path: str | None = None


@dataclass
class TeacherCacheConfig:
    enabled: bool
    cache_dir: str
    cache_dtype: str
    cache_layers: str
    cache_layer_interval: int
    cache_router_logits: bool


@dataclass
class DataConfig:
    train_file: str
    max_seq_len: int
    max_samples: int | None
    val_split: float = 0.0


@dataclass
class QuantizationConfig:
    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str


@dataclass
class TrainableConfig:
    embedding: bool
    lm_head: bool
    attention: Literal["freeze", "lora", "full"]
    shared_expert: Literal["freeze", "lora", "full"]
    routed_expert: Literal["freeze", "lora", "full"]
    router: Literal["freeze", "full"]


@dataclass
class LoRAConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


@dataclass
class LossesConfig:
    hidden_mse: float
    router_kl: float
    sft_ce: float
    hidden_layer_weighting: str
    router_kl_temperature: float


@dataclass
class LRSchedulerConfig:
    type: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = 0.1


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = "./outputs/tb"


@dataclass
class LayerwiseLoRAConfig:
    """LoRA + optional 4bit settings for layerwise training.

    When ``enabled`` is true, attention sub-modules (full + linear attention
    Linears matched by ``target_modules``) are loaded as base + LoRA adapter.
    With ``load_in_4bit=True`` the base weight is quantized via bitsandbytes
    Linear4bit; only the LoRA A/B matrices and non-LoRA, non-norm parameters
    train. All norm parameters in the layer are frozen regardless.
    """

    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    )
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class LayerwiseConfig:
    enabled: bool
    max_steps_per_block: int
    mse_threshold: float
    patience: int
    learning_rate: float
    optimizer: Literal[
        "adamw_8bit",
        "paged_adamw_8bit",
        "adamw_fp32",
        "sso",
        "sphere",
        "muon",
        "muon_triton",
        "muon_triton_batched",
    ]
    output_dir: str
    use_router_kl: bool
    save_every_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    sso_ns_steps: int
    sso_radius_c: float
    sso_radius_mode: str
    sso_msign_dtype: str
    sso_bisect_max_iters: int
    sso_bisect_tol: float
    sso_power_iters: int
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_paged_momentum: bool = False
    lr_scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.0
    log_every_steps: int = 20
    eval_every_steps: int = 0
    gradient_checkpointing: bool = True
    use_student_rollout_input: bool = False
    lora: LayerwiseLoRAConfig = field(default_factory=LayerwiseLoRAConfig)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    gradient_checkpointing: bool
    use_flash_attention: bool
    seed: int
    save_steps: int
    output_dir: str
    quantization: QuantizationConfig
    trainable: TrainableConfig
    lora: LoRAConfig
    losses: LossesConfig
    layerwise: LayerwiseConfig
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    eval_steps: int = 0


_DEFAULT_GGUF_QUANTS: tuple[str, ...] = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
_ALLOWED_GGUF_QUANTS: frozenset[str] = frozenset({
    *_DEFAULT_GGUF_QUANTS,
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q5_K_S",
    "BF16",
    "F16",
})


_ALLOWED_MMPROJ_OUTTYPES: tuple[str, ...] = ("bf16", "f16", "f32")


@dataclass
class ExportGGUFConfig:
    """Settings for `scripts/export_gguf.py` (safetensor → quantized GGUF)."""

    input_dir: str
    output_dir: str
    llama_cpp_dir: str | None = None
    llama_cpp_src_dir: str | None = None
    quant_types: list[str] = field(default_factory=lambda: list(_DEFAULT_GGUF_QUANTS))
    drop_mtp: bool = True
    keep_bf16: bool = False
    work_dir: str | None = None
    smoke_test: bool = True
    export_mmproj: bool = True
    mmproj_outtype: Literal["bf16", "f16", "f32"] = "bf16"


@dataclass
class AppConfig:
    project: ProjectConfig
    download: DownloadConfig
    prune: PruneConfig
    teacher_cache: TeacherCacheConfig
    data: DataConfig
    train: TrainConfig
    export_gguf: ExportGGUFConfig | None = None
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_DEFAULT_LAYERWISE_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def _parse_layerwise_lora(raw: dict[str, Any]) -> LayerwiseLoRAConfig:
    targets = raw.get("target_modules")
    if targets is None:
        target_modules = _DEFAULT_LAYERWISE_LORA_TARGETS
    else:
        target_modules = tuple(str(x) for x in targets)
    return LayerwiseLoRAConfig(
        enabled=bool(raw.get("enabled", False)),
        r=int(raw.get("r", 16)),
        alpha=int(raw.get("alpha", 32)),
        dropout=float(raw.get("dropout", 0.05)),
        target_modules=target_modules,
        load_in_4bit=bool(raw.get("load_in_4bit", True)),
        bnb_4bit_compute_dtype=str(raw.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=str(raw.get("bnb_4bit_quant_type", "nf4")),
    )


def _parse_export_gguf(raw: dict[str, Any] | None) -> ExportGGUFConfig | None:
    if not raw:
        return None
    quants_raw = raw.get("quant_types")
    if quants_raw is None:
        quants = list(_DEFAULT_GGUF_QUANTS)
    else:
        quants = [str(x) for x in quants_raw]
    bad = [q for q in quants if q not in _ALLOWED_GGUF_QUANTS]
    if bad:
        raise ValueError(
            f"export_gguf.quant_types contains unsupported entries {bad}. "
            f"Allowed: {sorted(_ALLOWED_GGUF_QUANTS)}"
        )
    mmproj_outtype = str(raw.get("mmproj_outtype", "bf16")).lower()
    if mmproj_outtype not in _ALLOWED_MMPROJ_OUTTYPES:
        raise ValueError(
            f"export_gguf.mmproj_outtype must be one of {list(_ALLOWED_MMPROJ_OUTTYPES)}; "
            f"got {raw.get('mmproj_outtype')!r}"
        )
    return ExportGGUFConfig(
        input_dir=str(_require(raw, "input_dir", "export_gguf")),
        output_dir=str(_require(raw, "output_dir", "export_gguf")),
        llama_cpp_dir=(str(raw["llama_cpp_dir"]) if raw.get("llama_cpp_dir") else None),
        llama_cpp_src_dir=(str(raw["llama_cpp_src_dir"]) if raw.get("llama_cpp_src_dir") else None),
        quant_types=quants,
        drop_mtp=bool(raw.get("drop_mtp", True)),
        keep_bf16=bool(raw.get("keep_bf16", False)),
        work_dir=(str(raw["work_dir"]) if raw.get("work_dir") else None),
        smoke_test=bool(raw.get("smoke_test", True)),
        export_mmproj=bool(raw.get("export_mmproj", True)),
        mmproj_outtype=mmproj_outtype,  # type: ignore[arg-type]
    )


def load_config(path: str | Path) -> AppConfig:
    raw = load_yaml(path)
    proj = _require(raw, "project", "root")
    dl = _require(raw, "download", "root")
    pr = _require(raw, "prune", "root")
    tc = raw.get("teacher_cache") or {}
    data = _require(raw, "data", "root")
    tr = _require(raw, "train", "root")
    eg = raw.get("export_gguf")

    project = ProjectConfig(
        name=str(_require(proj, "name", "project")),
        output_dir=str(_require(proj, "output_dir", "project")),
    )
    download = DownloadConfig(
        model_id=str(_require(dl, "model_id", "download")),
        revision=str(dl.get("revision", "main")),
        hf_endpoint=dl.get("hf_endpoint"),
        local_dir=str(_require(dl, "local_dir", "download")),
    )
    prune = PruneConfig(
        target_num_experts=int(_require(pr, "target_num_experts", "prune")),
        target_num_experts_per_tok=int(_require(pr, "target_num_experts_per_tok", "prune")),
        keep_shared_experts=bool(pr.get("keep_shared_experts", True)),
        expert_selection=str(pr.get("expert_selection", "first_n")),  # type: ignore[arg-type]
        manual_experts=pr.get("manual_experts"),
        router_stat_samples=int(pr.get("router_stat_samples", 500)),
        student_dir=str(_require(pr, "student_dir", "prune")),
        router_stats_path=(str(pr["router_stats_path"]) if pr.get("router_stats_path") else None),
    )
    if prune.expert_selection not in ("first_n", "router_top", "manual"):
        raise ValueError("prune.expert_selection must be first_n | router_top | manual")

    teacher_cache = TeacherCacheConfig(
        enabled=bool(tc.get("enabled", False)),
        cache_dir=str(tc.get("cache_dir", "./cache/teacher_hiddens")),
        cache_dtype=str(tc.get("cache_dtype", "float16")),
        cache_layers=str(tc.get("cache_layers", "every_4")),
        cache_layer_interval=int(tc.get("cache_layer_interval", 4)),
        cache_router_logits=bool(tc.get("cache_router_logits", True)),
    )
    dcfg = DataConfig(
        train_file=str(_require(data, "train_file", "data")),
        max_seq_len=int(_require(data, "max_seq_len", "data")),
        max_samples=data.get("max_samples"),
        val_split=float(data.get("val_split", 0.0)),
    )
    if dcfg.max_samples is not None:
        dcfg.max_samples = int(dcfg.max_samples)
    if not (0.0 <= dcfg.val_split < 0.5):
        raise ValueError("data.val_split must be in [0, 0.5)")

    q = _require(tr, "quantization", "train")
    trainable = _require(tr, "trainable", "train")
    lora = _require(tr, "lora", "train")
    losses = _require(tr, "losses", "train")
    layerwise = tr.get("layerwise") or {}
    lr_sched_raw = tr.get("lr_scheduler") or {}
    tb_raw = tr.get("tensorboard") or {}

    train = TrainConfig(
        epochs=int(_require(tr, "epochs", "train")),
        batch_size=int(_require(tr, "batch_size", "train")),
        gradient_accumulation_steps=int(_require(tr, "gradient_accumulation_steps", "train")),
        learning_rate=float(_require(tr, "learning_rate", "train")),
        warmup_ratio=float(tr.get("warmup_ratio", 0.0)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        max_grad_norm=float(tr.get("max_grad_norm", 1.0)),
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", False)),
        use_flash_attention=bool(tr.get("use_flash_attention", False)),
        seed=int(tr.get("seed", 42)),
        save_steps=int(tr.get("save_steps", 500)),
        output_dir=str(_require(tr, "output_dir", "train")),
        quantization=QuantizationConfig(
            load_in_4bit=bool(q.get("load_in_4bit", True)),
            bnb_4bit_compute_dtype=str(q.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_quant_type=str(q.get("bnb_4bit_quant_type", "nf4")),
        ),
        trainable=TrainableConfig(
            embedding=bool(trainable.get("embedding", False)),
            lm_head=bool(trainable.get("lm_head", False)),
            attention=str(trainable.get("attention", "lora")),  # type: ignore[arg-type]
            shared_expert=str(trainable.get("shared_expert", "lora")),  # type: ignore[arg-type]
            routed_expert=str(trainable.get("routed_expert", "lora")),  # type: ignore[arg-type]
            router=str(trainable.get("router", "full")),  # type: ignore[arg-type]
        ),
        lora=LoRAConfig(
            r=int(lora.get("r", 16)),
            alpha=int(lora.get("alpha", 32)),
            dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", [])),
        ),
        losses=LossesConfig(
            hidden_mse=float(losses.get("hidden_mse", 1.0)),
            router_kl=float(losses.get("router_kl", 0.5)),
            sft_ce=float(losses.get("sft_ce", 0.3)),
            hidden_layer_weighting=str(losses.get("hidden_layer_weighting", "uniform")),
            router_kl_temperature=float(losses.get("router_kl_temperature", 2.0)),
        ),
        layerwise=LayerwiseConfig(
            enabled=bool(layerwise.get("enabled", False)),
            max_steps_per_block=int(layerwise.get("max_steps_per_block", 2000)),
            mse_threshold=float(layerwise.get("mse_threshold", 1e-3)),
            patience=int(layerwise.get("patience", 400)),
            learning_rate=float(layerwise.get("learning_rate", 5e-5)),
            optimizer=str(layerwise.get("optimizer", "adamw_8bit")),  # type: ignore[arg-type]
            output_dir=str(layerwise.get("output_dir", "./models/student_layerwise")),
            use_router_kl=bool(layerwise.get("use_router_kl", True)),
            save_every_steps=int(layerwise.get("save_every_steps", 200)),
            batch_size=int(layerwise.get("batch_size", 1)),
            gradient_accumulation_steps=int(layerwise.get("gradient_accumulation_steps", 1)),
            sso_ns_steps=int(layerwise.get("sso_ns_steps", 6)),
            sso_radius_c=float(layerwise.get("sso_radius_c", 1.0)),
            sso_radius_mode=str(layerwise.get("sso_radius_mode", "preserve")),
            sso_msign_dtype=str(layerwise.get("sso_msign_dtype", "fp32")),
            sso_bisect_max_iters=int(layerwise.get("sso_bisect_max_iters", 20)),
            sso_bisect_tol=float(layerwise.get("sso_bisect_tol", 2e-4)),
            sso_power_iters=int(layerwise.get("sso_power_iters", 4)),
            muon_momentum=float(layerwise.get("muon_momentum", 0.95)),
            muon_ns_steps=int(layerwise.get("muon_ns_steps", 5)),
            muon_paged_momentum=bool(layerwise.get("muon_paged_momentum", False)),
            lr_scheduler_type=str(layerwise.get("lr_scheduler_type", "cosine")),  # type: ignore[arg-type]
            min_lr_ratio=float(layerwise.get("min_lr_ratio", 0.1)),
            warmup_ratio=float(layerwise.get("warmup_ratio", 0.0)),
            log_every_steps=int(layerwise.get("log_every_steps", 20)),
            eval_every_steps=int(layerwise.get("eval_every_steps", 0)),
            gradient_checkpointing=bool(layerwise.get("gradient_checkpointing", True)),
            use_student_rollout_input=bool(layerwise.get("use_student_rollout_input", False)),
            lora=_parse_layerwise_lora(layerwise.get("lora") or {}),
        ),
        lr_scheduler=LRSchedulerConfig(
            type=str(lr_sched_raw.get("type", "cosine")),  # type: ignore[arg-type]
            min_lr_ratio=float(lr_sched_raw.get("min_lr_ratio", 0.1)),
        ),
        tensorboard=TensorBoardConfig(
            enabled=bool(tb_raw.get("enabled", True)),
            log_dir=str(tb_raw.get("log_dir", "./outputs/tb")),
        ),
        eval_steps=int(tr.get("eval_steps", 0)),
    )

    for name, mode in (
        ("attention", train.trainable.attention),
        ("shared_expert", train.trainable.shared_expert),
        ("routed_expert", train.trainable.routed_expert),
    ):
        if mode not in ("freeze", "lora", "full"):
            raise ValueError(f"train.trainable.{name} must be freeze|lora|full")
    if train.trainable.router not in ("freeze", "full"):
        raise ValueError("train.trainable.router must be freeze|full")
    if train.layerwise.optimizer not in (
        "adamw_8bit",
        "paged_adamw_8bit",
        "adamw_fp32",
        "sso",
        "sphere",
        "muon",
        "muon_triton",
        "muon_triton_batched",
    ):
        raise ValueError(
            "train.layerwise.optimizer must be adamw_8bit | paged_adamw_8bit | "
            "adamw_fp32 | sso | sphere | muon | muon_triton | muon_triton_batched"
        )
    if train.layerwise.batch_size < 1:
        raise ValueError("train.layerwise.batch_size must be >= 1")
    if train.layerwise.gradient_accumulation_steps < 1:
        raise ValueError("train.layerwise.gradient_accumulation_steps must be >= 1")
    if train.lr_scheduler.type not in ("cosine", "linear", "constant"):
        raise ValueError("train.lr_scheduler.type must be cosine | linear | constant")
    if not (0.0 <= train.lr_scheduler.min_lr_ratio <= 1.0):
        raise ValueError("train.lr_scheduler.min_lr_ratio must be in [0, 1]")
    if train.layerwise.lr_scheduler_type not in ("cosine", "linear", "constant"):
        raise ValueError("train.layerwise.lr_scheduler_type must be cosine | linear | constant")
    if not (0.0 <= train.layerwise.min_lr_ratio <= 1.0):
        raise ValueError("train.layerwise.min_lr_ratio must be in [0, 1]")
    if not (0.0 <= train.layerwise.warmup_ratio < 1.0):
        raise ValueError("train.layerwise.warmup_ratio must be in [0, 1)")
    if train.layerwise.log_every_steps < 1:
        raise ValueError("train.layerwise.log_every_steps must be >= 1")
    if train.layerwise.eval_every_steps < 0:
        raise ValueError("train.layerwise.eval_every_steps must be >= 0")
    if train.eval_steps < 0:
        raise ValueError("train.eval_steps must be >= 0")
    if train.layerwise.lora.enabled:
        if train.layerwise.lora.r < 1:
            raise ValueError("train.layerwise.lora.r must be >= 1")
        if train.layerwise.lora.alpha < 1:
            raise ValueError("train.layerwise.lora.alpha must be >= 1")
        if not (0.0 <= train.layerwise.lora.dropout < 1.0):
            raise ValueError("train.layerwise.lora.dropout must be in [0, 1)")
        if not train.layerwise.lora.target_modules:
            raise ValueError(
                "train.layerwise.lora.target_modules must be non-empty when enabled"
            )
        if train.layerwise.lora.bnb_4bit_compute_dtype.lower() not in (
            "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"
        ):
            raise ValueError(
                "train.layerwise.lora.bnb_4bit_compute_dtype must be one of "
                "bf16/bfloat16/fp16/float16/fp32/float32"
            )
        if train.layerwise.lora.bnb_4bit_quant_type not in ("nf4", "fp4"):
            raise ValueError("train.layerwise.lora.bnb_4bit_quant_type must be nf4 | fp4")

    cfg = AppConfig(
        project=project,
        download=download,
        prune=prune,
        teacher_cache=teacher_cache,
        data=dcfg,
        train=train,
        export_gguf=_parse_export_gguf(eg),
        raw=raw,
    )
    validate_paths(cfg)
    return cfg


def validate_paths(cfg: AppConfig) -> None:
    if cfg.prune.target_num_experts < 1:
        raise ValueError("prune.target_num_experts must be >= 1")
    if cfg.prune.target_num_experts_per_tok < 1:
        raise ValueError("prune.target_num_experts_per_tok must be >= 1")
    if cfg.prune.target_num_experts_per_tok > cfg.prune.target_num_experts:
        raise ValueError("target_num_experts_per_tok cannot exceed target_num_experts")
    if cfg.prune.expert_selection == "manual" and not cfg.prune.manual_experts:
        raise ValueError("manual_experts required when expert_selection=manual")
