# image_bricks

整合VAGEN和Isaac的项目，用于brick相关的训练和评估。

## 项目结构

```
image_bricks/
├── examples/          # 你的脚本和main函数，会同时调用Isaac和VAGEN
├── VAGEN/            # VAGEN开源项目（保持原样）
├── Isaac/            # Isaac开源项目（保持原样）
└── configs/          # 配置文件
```

## 安装步骤

### 1. 克隆项目并初始化submodules

```bash
git clone <your-repo-url>
cd image_bricks
git submodule update --init --recursive
```

### 2. 安装VAGEN

```bash
cd VAGEN
pip install -e .
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ../..
pip install "trl==0.26.2"
```

### 3. 安装Isaac

```bash
cd Isaac
pip install -e .  # 根据Isaac项目的实际setup.py调整
cd ..
```

### 4. 安装根项目（可选）

```bash
pip install -e .
```

## 使用方式

在`examples/`目录下的脚本可以直接导入VAGEN和Isaac：

```python
# 导入VAGEN
from vagen.envs.brick import BrickBuilder
from vagen.main_ppo import run_ppo

# 导入Isaac（根据实际包名调整）
# from isaac.sim import IsaacSim

# 在你的代码中组合使用
```

## 注意事项

1. **保持VAGEN和Isaac独立**：这两个是开源项目，保持它们的目录结构不变，便于后续更新
2. **Submodule管理**：`.gitmodules`在根目录统一管理所有submodules
3. **导入路径**：确保VAGEN和Isaac都已安装（`pip install -e .`），这样Python才能找到这些包

## 开发建议

- 在`examples/`中编写你的主要代码
- 使用`configs/`存放配置文件
- 保持VAGEN和Isaac目录的原样，不要直接修改它们的代码（除非fork）