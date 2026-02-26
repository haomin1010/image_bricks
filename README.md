# image_bricks



## Installation

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