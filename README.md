# image_bricks v2.2

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

## Data Generate

Now, we use scripts in data_genv3 to finish data generate, you can generate pic, data via

```bash
python IsaacLab/scripts/data_genv3/step1_render_scatter_build_sequence_v2.py   --output_root assets/dataset_v3/smallsize_4   --output_count 2

./IsaacLab/isaaclab.sh -p IsaacLab/scripts/data_genv3/step2_render_from_struct_v3.py   --enable_cameras   --output_root assets/dataset_v3/smallsize_4   --output_count 2   --headless
```

//TODO：We need to finish SFT data generate and CoT data collection scripts.

## Train 

### 1. train with grpo

```
bash VAGEN/examples/isaac/train_grpo_qwen25vl3b.sh
```

## Evaluate

### 1. BrickIsaac + Qwen


```bash
bash scripts/eval_isaac.sh
```

### 2. BrickIsaac + OpenRouter

```bash
bash scripts/eval_isaac_openrouter.sh
```

You Can change args in `scripts/template.sh`

