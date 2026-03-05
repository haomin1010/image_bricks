# image_bricks v1.0

## Installation

Before getting into the following steps, make sure that CUDA Toolkit and Vulkan are set properly.

### 0. create a conda environment

```bash
conda create -n bricks python==3.11
conda activate bricks

```

### 1. install isaaclab

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
cd Isaaclab
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

### 2. install VAGEN

```bash
cd VAGEN
pip install -e .
git submodule update --init --recursive
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ../..
pip install "trl==0.26.2"
```

### 3. install numpy & flash_attn

```bash
pip install numpy==1.26.4
pip install flash_attn==2.8.3 --no-build-isolation
```

## Train 

### 1. train with grpo

```
bash VAGEN/examples/isaac/train_grpo_qwen25vl3b.sh
```

## Evaluate

### 1. evaluate with qwen-VL3 API

```
bash VAGEN/examples/evaluate/isaac/run_eval.sh
```

If failed, try:

```
ISAAC_CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 ISAAC_HEADLESS=1 bash VAGEN/examples/evaluate/isaac/run_eval.sh
```

# image_bricks v2.0

**太长不看版：在 scripts/ 中新建一个 .sh 文件，调用 scripts/template.sh 模板，指定所有参数。可以参考已有的 eval_isaac.sh。**

重构后：不改动 `IsaacLab/`，不改变 `VAGEN/` 的总体结构（不搬家、不重排目录），主要目标是**把 evaluate 相关的可配置参数统一到仓库根目录 `scripts/*.sh`**，避免在多个脚本 / Python / 环境变量里重复定义，导致“不知道该改哪里”的问题。

---

## 启动 evaluate

### 1) BrickIsaac + Qwen（默认）

在仓库根目录执行：

```bash
bash scripts/eval_isaac.sh
```

它等价于过去常见的启动方式（在环境变量里指定 Isaac 相关参数，再执行 `VAGEN/examples/evaluate/isaac/run_eval.sh`），只是现在默认值与可改参数都集中在 `scripts/template.sh`。

### 2) BrickIsaac + OpenRouter（OpenAI 兼容）

```bash
bash scripts/eval_isaac_openrouter.sh
```

---

## 旧入口是否还能用？

可以。以下命令仍然可用（并且会转而从 `scripts/template.sh` 读取默认配置）：

```bash
ISAAC_CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 ISAAC_HEADLESS=1 bash VAGEN/examples/evaluate/isaac/run_eval.sh
```

以及：

```bash
bash VAGEN/examples/evaluate/isaac/run_eval_openrouter.sh
```

---

## 参数统一在哪？

所有 evaluate 相关的默认值与参数接口集中在：

- `scripts/template.sh`

你可以用两种方式覆写参数：

1. **导出环境变量**（推荐，最直观，适合 GPU/密钥/路径等）
2. **在命令行追加 OmegaConf dotlist overrides**（与原来一致，适合改 YAML 配置项）

---

## 常用配置项（环境变量）

下面的变量在 `scripts/template.sh` 中定义默认值，并由脚本导出给 Python 使用。

### 1) Isaac / GPU 相关

- `ISAAC_CUDA_VISIBLE_DEVICES`：给 Isaac Sim 子进程用的 `CUDA_VISIBLE_DEVICES`（默认 `0`）
- `DEVICE`：Isaac device（默认 `cuda:0`）
- `ISAAC_HEADLESS`：是否 headless（默认 `1`；你也可以设为 `0`）

示例：

```bash
ISAAC_CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:1 ISAAC_HEADLESS=1 bash scripts/eval_isaac.sh
```

### 2) Isaac Server 启动参数（由 `VAGEN/vagen/evaluate/run_eval.py` 读取）

这些参数原来在 `run_eval.py` 中部分硬编码（例如 task/num-envs），现在统一改为从环境变量读取：

- `ISAAC_SERVER_NUM_ENVS`：传给 `start_isaac_server.py --num-envs`（默认 `1`）
- `ISAAC_SERVER_TASK`：传给 `--task`（默认 `Isaac-Stack-Cube-UR10-Short-Suction-IK-Rel-v0`）
- `ISAAC_SERVER_RECORD`：`1` 开启 `--record`（默认 `0`）
- `ISAAC_SERVER_VIDEO_LENGTH`：传给 `--video-length`（默认 `0` 表示不限制）
- `ISAAC_SERVER_VIDEO_INTERVAL`：传给 `--video-interval`（默认 `0`）
- `ISAAC_SERVER_IK_LAMBDA_VAL`：传给 `--ik-lambda-val`（默认空，不传）
- `ISAAC_SERVER_EXTRA_ARGS`：附加原始参数（会 `shlex.split`），例如：`"--no-headless --task xxx"`

示例（改 task + 开录制）：

```bash
ISAAC_SERVER_TASK="multipicture_assembling_from_begin" \
ISAAC_SERVER_RECORD=1 \
bash scripts/eval_isaac.sh
```

### 3) LLM 后端与密钥

**重要：本仓库不再在脚本里硬编码任何 API KEY。** 请在你自己的环境里设置密钥（例如写进 `~/.bashrc` 或 CI secrets）。

Qwen（默认）：

- `QWEN_API_KEY`：必需（不设置会报错）
- `QWEN_BASE_URL`：默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`

OpenRouter（OpenAI 兼容客户端）：

- `OPENAI_API_KEY`：必需
- `OPENROUTER_BASE_URL`：默认 `https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL_ID`：默认 `qwen/qwen3.5-flash-02-23`

### 4) 输出路径 / fileroot

- `VAGEN_FILEROOT`：默认 `${REPO_ROOT}/VAGEN`

注意：当前 `VAGEN/examples/evaluate/isaac/config.yaml` 内部的 `fileroot` 仍然是历史绝对路径；为了不改动上游结构，我们继续通过脚本注入 `fileroot=...` 覆写它（行为与旧脚本一致）。

### 5) 运行时清理（保持旧逻辑，必要时可关）

旧脚本在每次启动前会执行：

- `ray stop --force`
- `pkill -f start_isaac_server.py`

现在这两个动作统一由 `scripts/template.sh` 执行，并提供开关：

- `VAGEN_EVAL_RAY_STOP_FORCE=0`：不自动 stop ray
- `VAGEN_EVAL_KILL_OLD_ISAAC_SERVER=0`：不自动 pkill Isaac server

---

## OmegaConf overrides（与原来一致）

你仍然可以像以前一样在命令末尾追加 overrides，例如：

```bash
bash scripts/eval_isaac.sh run.max_concurrent_jobs=2 experiment.dump_dir=./rollouts/tmp
```

如果你显式传入了 `run.backend=...`，脚本会尊重你的选择，并**不会**再注入默认后端（这与旧的 `VAGEN/examples/evaluate/isaac/run_eval.sh` 行为一致）。

---

## 变更点总览（给维护者/合作者）

- 新增统一配置入口：`scripts/template.sh`
- 新增推荐入口脚本：
  - `scripts/eval_isaac.sh`
  - `scripts/eval_isaac_openrouter.sh`
- 保持原有入口脚本路径不变，仅改为：
  - `VAGEN/examples/evaluate/isaac/run_eval.sh`：source `scripts/template.sh` 后调用统一入口
  - `VAGEN/examples/evaluate/isaac/run_eval_openrouter.sh`：同上
- `VAGEN/vagen/evaluate/run_eval.py`：把 Isaac server 的 `--num-envs/--task/...` 等参数从硬编码改为读取环境变量（由 `scripts/template.sh` 提供默认值）

---

## 排错建议

- 报 “API key missing”：检查 `QWEN_API_KEY` 或 `OPENAI_API_KEY` 是否已导出。
- Isaac 启动失败 / 找不到 GPU：检查 `ISAAC_CUDA_VISIBLE_DEVICES`、`DEVICE` 是否匹配你的机器。
- 需要 GUI：设 `ISAAC_HEADLESS=0`（并保证有可用的 `DISPLAY`）。
- 不想每次都 stop ray：设 `VAGEN_EVAL_RAY_STOP_FORCE=0`。

