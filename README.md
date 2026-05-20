# MoE Prune & Distill

针对 Qwen3.5 MoE 系列大模型的 **结构化专家剪枝 + LoRA 蒸馏修复** 工具链，
目标硬件单张 16GB GPU。详细设计见 [DESIGN.md](DESIGN.md)。

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

依赖 `transformers>=5.x`（含 `qwen3_5_moe` 模型）、`bitsandbytes`、`peft`、
`accelerate`、`safetensors`。**Linux / WSL2 推荐**：Windows 上 `bitsandbytes`
可能不可用。

## 端到端流程（推荐）

```bash
python -m scripts.download           --config configs/example.yaml
python -m scripts.inspect            --config configs/example.yaml
python -m scripts.build_train_set    --config configs/data_sources.yaml --smoke   # 验证
python -m scripts.build_train_set    --config configs/data_sources.yaml           # 正式
python -m scripts.stream_teacher     --config configs/example.yaml                # router stats + teacher cache (合并)
python -m scripts.prune              --config configs/example.yaml
# 或者：保留剪掉专家的权重，按相似度融合到留存专家里再走后续步骤
# python -m scripts.prune_merge      --config configs/example.yaml --merge-strategy weight_cosine --merge-alpha 0.5
python -m scripts.train_layerwise    --config configs/example.yaml                # 逐 block 蒸馏 (新)
python -m scripts.train              --config configs/example.yaml \
    --student-dir-override ./models/student_layerwise --offload-folder ./cache/offload
```

每步说明：

- **download**：设 `HF_ENDPOINT`，`snapshot_download` 到 `download.local_dir`。
- **inspect**：读教师 `config.json`，打印架构 / 层数 / 专家数 / 参数量粗估。
- **build_train_set**：从多源 HF 数据集流式拉取，按 token 配额合成
  `data/train.jsonl`。`--smoke` 给每源 500 tokens 上限，全套约半分钟跑完。
- **stream_teacher**：`collect_router_stats + cache_teacher` 的等价快速路径
  （DESIGN §7.3）。按 layer 走全数据集，把 35B 教师从 `device_map=auto` +
  磁盘 offload 解放出来；输出 `router_stats.json` 与教师 cache。**Cache
  schema v2（2026-05-17）**：每个被缓存的 layer 切成若干 chunk
  `cache_layer_{i}_chunk_{j}.safetensors`（每 chunk 装 `--chunk-size`
  个样本，默认 500），`cache_meta.safetensors` 装所有样本的
  `input_ids` / `attention_mask`，`cache_index.json` 记录 sid → chunk 映射。
  下游 `load_sample_cache` 自动在 v1（`{sid}.safetensors`）/ v2 之间分流，
  `DistillJsonlDataset` / `layerwise_trainer` 不需要改。砍掉了原
  `append_sample_cache` 的写放大 + finalize 阶段的额外读+写，cache 写盘
  I/O 从 ~10TB 降到 ~530GB（**~20× 削减**）。`--skip-existing` 的语义改为
  按 `cache_index.json` 的 sample 集合判断；如果中途崩溃要重跑，删
  `cache_index.json` + `cache_layer_*_chunk_*.safetensors` 后再启动。
  **`--batch-size`（默认 4）**：每个 layer 同时 forward N 个样本，按 seq
  长度排序后右 padding 到 batch 内最长，attention_mask 把 pad 位从
  attention / router top-k 计数里抠掉。等价于 B=1（fp32 下逐样本输出
  bit-exact，bf16/fp16 下差异 < 1e-3）；GPU kernel launch overhead 摊
  到 N 个样本上，B=4 通常 2–3× compute 加速。RAM 紧或 OOM 调到
  `--batch-size 1` 即可退回原路径。
- **prune**：按 `prune.expert_selection`（`first_n` / `router_top` / `manual`）
  生成剪枝学生，写出 `expert_mapping.json`。**流式写盘**：逐 shard `safe_open`
  → slice → `save_file` → free，内存峰值 ≈ 单 shard (≤ 6GB)。
- **prune_merge**（可选替代 prune）：剪枝时不直接扔掉被踢出的专家，
  按相似度把它们的权重以加权平均的方式融到留存专家里
  （`W_kept_new[k] = W_kept[k] + alpha · Σ_d w[d,k] · W_dropped[d]`）。
  策略三选：
  - `weight_cosine`（默认）：对每个专家把 `gate_up_proj + down_proj`
    拍平后求 cosine 相似度，再 `softmax(sim/tau)` 得到权重。最贴近"完整
    功能相似度"，但一层 ~7GB fp32 内存峰值。
  - `weight_cosine_of_router`：用 router gate 矩阵中每个专家对应的那一
    行（`mlp.gate.weight[e]`）当作专家的特征向量做 cosine。一层只读
    ~4MB（256 专家 × 4096 hidden × 4B），等价于"教师 router 自己怎么
    把专家分开就怎么融合"。便宜得多，与 `weight_cosine` 互补。
  - `cooccur`：读 `router_cooccur.json` 的 top-k 共现频率，v1 该收集脚本
    仍是 stub。
  输出与 `prune` 完全兼容（同 shard 布局、同 `expert_mapping.json`），
  额外写一份 `merge_plan.json` 记录每层 `[d_id, k_id, weight]` 三元组
  便于审计。`--merge-alpha 0` 等价于纯 `prune`。Router gate 不参与融合
  （按行 index_select）。详见
  [moe_prune_distill/prune/expert_merge.py](moe_prune_distill/prune/expert_merge.py)。
- **train_layerwise**：按 `cache_layers=every_4` 把 40 层切成 10 个 block，
  每 block 只把对应 1–4 层的 bf16 副本驻 GPU；用现有 teacher cache 的
  `hidden.layer_{0,4,8,...,36}` 做 input/target，`normalized_hidden_mse` +
  可选 `router_kl` 训到 EMA < `mse_threshold` 或触发 patience。每 block
  独立可断点续跑（`block_*.done.json` 标记）。完成后 merge 到
  `models/student_layerwise/`，shard 布局与 `prune` 输出一致。GPU 峰值 < 8GB。
  可选 `train.layerwise.batch_size` / `gradient_accumulation_steps`
  控制单步样本量与累计步数（默认 `1/1`，即逐样本更新）；CLI 同名 flag
  覆盖 YAML。block 之间显式 `del optimizer` + `empty_cache` 避免状态残留，
  详见 [memory/2026-05-17.md](memory/2026-05-17.md)。
  可选 `train.layerwise.use_student_rollout_input`（默认 `false`）：开启后
  block N 训完会再做一遍 eval 前向，把学生 hidden 写到
  `_layers/_rollout/`；block N+1 的 *输入* 改读这份 rollout cache（loss
  target 仍是 teacher cache）。这在训练时引入 N→N+1 的耦合，与推理时
  误差累积路径对齐。代价是每 block 多 5–10 分钟前向 + ~`1/cache_layers`
  ×teacher 体积的盘空间。开启时会强制 block 顺序执行（`--blocks` 必须
  是连续区间）。CLI: `--use-student-rollout-input` /
  `--no-use-student-rollout-input` 覆盖 YAML。
  可选 `optimizer ∈ {adamw_8bit, adamw_fp32, sso, sphere, muon, muon_triton, muon_triton_batched}`：
  SSO 谱球优化器在剪枝-蒸馏场景下要把 `sso_radius_mode` 留在默认
  `preserve`（保留每个权重的 σ_init 当谱半径），切到论文默认的
  `paper` 公式会摧毁预训练特征并在死专家上触发 NaN，详见
  [memory/2026-05-17.md](memory/2026-05-17.md) §4。
  `muon_triton` / `muon_triton_batched` 走 Triton 加速的 Muon：NS5 内层
  `X @ X.T` 用对称化 kernel；`_batched` 把同形矩阵分桶批量做 NS5，**3D
  MoE expert stack `[E, M, N]` 直接喂给批量 NS5**（partition 改造前曾
  fall back 到 AdamW，覆盖率从 ~10% 提到 ~91% 的层参数）。需要 Triton
  （Linux: `pip install triton`；Windows: `pip install triton-windows`）。
- **train**：4bit + LoRA 蒸馏（hidden MSE + router KL + SFT CE），用
  `--student-dir-override` 把基座切到 layerwise 后的检查点。检测到
  teacher cache 即切到蒸馏模式，否则回落 P0 SFT。可选
  `--optimizer {adamw, muon_triton, muon_triton_batched}`，默认 `adamw`；
  Muon 模式把 LoRA A/B 与 router gate 等 2D 矩阵走 NS5，1D / embedding 走
  foreach AdamW。

旧路径仍可用作回退：

```bash
python -m scripts.collect_router_stats --config configs/example.yaml
python -m scripts.cache_teacher        --config configs/example.yaml --offload-folder ./cache/offload
```

测试：`pytest -q`（78 个用例：P0 + P1 + 流式等价性 + layerwise + 专家融合
+ SSO 优化器 + 指标/调度/val split）。

## 训练可观测性

`scripts/train.py` 与 `scripts/train_layerwise.py` 共用三套基础设施
（详见 [DESIGN.md §7.5](DESIGN.md)）：

**LR 调度**：cosine（默认）/ linear / constant + warmup。

```yaml
train:
  warmup_ratio: 0.03
  lr_scheduler:
    type: cosine            # cosine | linear | constant
    min_lr_ratio: 0.1       # final lr = lr * min_lr_ratio at total_steps
  layerwise:
    lr_scheduler_type: cosine    # 每个 block 独立重置
    min_lr_ratio: 0.1
    warmup_ratio: 0.03           # 占 max_steps_per_block 的比例
```

**验证集**：从 `train_file` 按 sample id 的 sha1 哈希做确定性 holdout，
端到端与 layerwise 共享同一份划分。

```yaml
data:
  val_split: 0.02         # 0 关闭；范围 [0, 0.5)

train:
  eval_steps: 200         # 端到端，0 -> 复用 save_steps
  layerwise:
    eval_every_steps: 0    # layerwise，0 -> 复用 log_every_steps
```

**指标 + JSONL 日志**：训练时 stdout/tqdm 同时按行追加到
`{output_dir}/train_log.jsonl`、`{output_dir}/val_log.jsonl`（layerwise
落到 `_layers/block_NNN_train_log.jsonl` / `_val_log.jsonl`）。

| 指标 | 频率 | 说明 |
| --- | --- | --- |
| `loss`, `hidden_mse`, `router_kl`, `sft_ce` | 每 step | 损失路径已算 |
| `ema_h` | 每 step | hidden_mse 的 EMA, β=0.95 |
| `lr`, `grad_norm`, `valid_tokens`, `mean_seq_len` | 每 step | 复用 `clip_grad_norm_` 返回值 |
| `nmse`, `cos_loss`, `teacher_norm`, `student_norm` | 每 eval | `‖s-t‖² / ‖t‖²`、`1 - cos(s,t)`、L2 范数 |
| `router_entropy` | 每 eval | 学生 raw logits 的熵，不带温度 |
| `removed_expert_mass` | 每 eval | `1 - Σ_surviving softmax(teacher_logits)`，仅在 teacher router cache + `expert_mapping.json` 都在场时输出 |

诊断 metric 在 `moe_prune_distill/distill/metrics.py` 里实现，全部
`@torch.no_grad`，与损失路径无 autograd 耦合。

## 训练数据合成

`scripts/build_train_set.py` 从 HF 流式拉取多源数据，配额单位是 **tokens**
（不是条数）。最终目标 200M tokens / `data_sources.yaml`。

```bash
# （可选，推荐）一次性预下慢启动源到 HF 缓存，后续 smoke 不再阻塞
python -m scripts.prefetch_datasets --config configs/data_sources.yaml \
  --only openhermes_2p5

# 烟雾测试：每源 500 tokens 上限
python -m scripts.build_train_set --config configs/data_sources.yaml --smoke

# 子集烟雾
python -m scripts.build_train_set --config configs/data_sources.yaml --smoke \
  --only fineweb_edu_zh,tulu3_sft

# 正式构建
python -m scripts.build_train_set --config configs/data_sources.yaml
```

数据源（`configs/data_sources.yaml`）覆盖：

- 中英日 web 文本：fineweb-edu-en/zh, allenai/c4-ja
- 代码 / 数学：CodeFeedback, open-web-math
- 通用 SFT：tulu3, smoltalk, belle-3.5M-CN, ShareGPT-zh-en, oasst1-ja, dolly-15k-ja
- 蒸馏风格：OpenHermes-2.5, ultrachat_200k
- 校正 SFT：no_robots, dolly-15k
- benchmark：MMLU(aux), GSM8K, HellaSwag
- VL（the_cauldron）：ai2d, chartqa, docvqa, ocrvqa

关键工程点：

- **HF_ENDPOINT 三层注入**：`moe_prune_distill/data/sources.py::_install_endpoint_redirect`
  打 httpx + requests + aiohttp 三层 monkeypatch，把所有出向请求重写到
  `hf-mirror.com`。`build_train_set.py` 在 tokenizer load **之前** 调用
  这个函数（必须早于 transformers 导入，否则 huggingface_hub 早期 session
  会把未重写的 URL 缓进连接池）。三层缺一不可：仅 httpx 时 datasets
  streaming 走 fsspec/requests 拉 shard 会回落到 `huggingface.co`
  → `scanned=0`；缺 aiohttp 时 parquet shard 直接 `ClientConnectorError`。
  开 `MPD_DEBUG_REDIRECT=1` 可打 patch 拦截到的每个 URL，定位漏网请求。
- **`parquet_glob` 旁路 loader**：`datasets` 自带 resolver 在 hf-mirror 下
  对脚本-based 库（fineweb-edu）和多 config / 多 split 的标准库
  （tulu3、smoltalk、mmlu、the_cauldron 等）会用错端点
  （`/resolve/<dir>/` 而非 `/api/.../tree/<dir>`），mirror 返回 404 →
  silent 空 IterableDataset。`extra: {loader: parquet_glob, subdir, glob}`
  改用 tree API 列文件 + 直接 `load_dataset("parquet", data_files=urls)`，
  跳过整条 auto-resolver。`format: json` 处理 jsonl/json.gz。
- **bucket_balance 软上限**：按 token 长度软分桶
  `[1,256] [256,512] [512,1024] [1024,2048]`，避免单一长度主导某源预算。
  上限是 `quota / n_buckets * 1.5`；当 `quota < bucket_edges[-1] *
  n_buckets`（默认 8192）时整体禁用——smoke 模式 quota=500 时 balance
  会卡到只接受 1 条样本，必须短路掉。
- **transforms 注册表**：`plain_text` / `messages_passthrough` /
  `instruction_io` / `conversations_sharegpt` / `conversations_humanassistant` /
  `mmlu_mc` / `hellaswag_mc` / `the_cauldron` / `vl_image_qa`。
- **dedup**：句首 256 字符 sha1；VL 记录再拼首图 32×32 缩略图的 sha1
  做"图文联合 key"——cauldron 模板化提示（ocrvqa 的"Who wrote this
  book?"）下纯文本 key 会把 165k 行折成 30 多条，必须靠图像区分。
- **VL 落盘**：图片转 JPEG 写到 `data/images/{sha16}.jpg`，jsonl 内是
  `{"type":"image","image":"file://..."}` 块。
- **快速失败**：连续 reject ≥ max(2000, quota/50) 行还没产出就放弃该源。
- **诊断工具**：
  - `scripts/probe_source.py <name>`：单源探测，`--reset-cache` 清缓存、
    `--no-streaming` 试整下载、`--raw-tree` 直接打镜像 tree API、
    `--scan-tree` 递归扫 repo 列每目录文件分布。镜像下任何源
    `scanned=0` 都从这条命令开始查。
  - `scripts/prefetch_datasets.py --only <name>`：调
    `huggingface_hub.snapshot_download` 一次性把指定源拉到本地缓存，
    给 OpenHermes-2.5 这类 6GB 单文件用。

镜像下踩坑全档案见 [memory/2026-05-17.md §6](memory/2026-05-17.md)。

## 状态 / 已知问题

- **测试**：78/78 通过（P0 7 + P1 24 含专家融合 6 + streamer 3 + layerwise 9
  + SSO 优化器 16 + 指标 10 + LR 调度 8）。在带 CUDA 的机器上额外 1 个 layerwise 收敛测试覆盖
  `batch_size>1 + grad_accum>1` 路径（CPU 上由 fla Triton kernel 跳过）。
  SSO 测试覆盖 NaN 兜底（tiny / zero / partial-dead-expert / NaN-safe
  bisection / preserve-radius 保留 σ_init / NaN-grad 跳过）。
- **prune 内存峰值**：流式重写后 ≤ 6GB（先前 30+GB），可在 16GB 物理内存机器跑通。
- **layerwise GPU 峰值**：4 层 bf16 + 8bit AdamW + checkpointing 下 < 8GB。
  先 layerwise 做 block-wise 对齐再走 train.py 端到端，比直接 4bit + LoRA 端到端
  起点显著更低。
- **流式 scratch 占盘**：`stream_teacher.py` 跑 3000 样本 / 2048 seq /
  hidden 4096 时 scratch 峰值 ≈ 50GB（chunked 布局，单倍水位）；2026-05-17
  v2 cache 改造后**没有额外 stage 目录**——cache 文件 streaming 期单调
  上升到 ≈ 530GB（= 最终 cache 体积，每张 (sample, layer) tensor 只写一
  次）。scratch 文件数 = `ceil(N_samples / chunk_size)`（默认
  `chunk_size=1000` → 10000 样本约 10 个 `scratch_chunk_*.cur.safetensors`
  文件，不再随样本数线性膨胀）。host RAM per-layer 峰值 ≈
  `chunk_size × seq × (hidden + num_experts) × 2 bytes` ≈ 17GB
  （chunk_size=1000），RAM 紧的可以下调 `--chunk-size`。最终 teacher
  cache 体积与旧 `cache_teacher.py` 相同，可通过减少 `cache_layers` /
  `max_samples` / `max_seq_len` 缩减。
- **数据源 mirror 兼容性**：`data_sources.yaml` 里 13 个源已切到
  `parquet_glob` 旁路。`c4_ja` 走 `multilingual/c4-ja*.json.gz`，mirror
  同步覆盖度不确定，跑前看 parquet_glob 报的 "matched files" 计数。
  `mmlu_aux/auxiliary_train` 的 parquet 把每行嵌进顶层 `train` struct
  （`{"train": {question, choices, answer, subject}}`），与 `all` config
  的 flat 行不同；`mmlu_mc` transform 入口已自动 unwrap，两种 layout 共用
  同一条 transform，详见 [memory/2026-05-17.md](memory/2026-05-17.md) §7。
- **bitsandbytes Windows**：4bit 加载 + AdamW8bit 在 Windows 上偶发，
  layerwise 自动 fallback 到 fp32 AdamW（多耗 ~2× 显存），train.py 4bit
  仍建议 Linux/WSL2。
- **transformers 版本**：`stream_teacher.py` / `train_layerwise.py` 需要
  `qwen3_5_moe` 模型，对应 `transformers>=5.x`。`cache_teacher.py` 路径同样依赖。
