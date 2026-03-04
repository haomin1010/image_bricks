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

## Test

### 1.test isaaclab env

```
python VAGEN/vagen/server/test_ray_grasp.py    --one-click-kill-existing-server true   --one-click-keep-ray true   --ray-head-log outputs/compare/ray_head.log   --server-num-envs 1   --server-device cuda:0   --server-task multipicture_assembling_from_begin   --server-record true   --server-video-length 0   --server-video-interval 0   --server-log outputs/compare/isaac_server_chain_test_headless.log   --seed 0   --goals "2,2,0;2,3,0;3,3,0" --server-headless true --auto-start true
```