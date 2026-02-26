"""
示例：如何在examples中同时使用VAGEN和Isaac

安装步骤：
1. cd VAGEN && pip install -e . && cd verl && pip install --no-deps -e . && cd ../..
2. cd Isaac && pip install -e . && cd ..
3. pip install -e .  # 安装根项目（可选）
"""

# 导入VAGEN的功能
from vagen.envs.brick import BrickBuilder
from vagen.envs.brick_isaac import BrickIsaac
# from vagen.main_ppo import run_ppo

# 导入Isaac的功能（根据实际Isaac项目的包结构调整）
# from isaac.sim import IsaacSim  # 示例，需要根据实际项目调整
# from isaac.utils import some_function

def main():
    """
    主函数：展示如何同时使用VAGEN和Isaac
    """
    print("示例：同时使用VAGEN和Isaac")
    
    # 使用VAGEN的环境
    # brick_env = BrickBuilder(...)
    
    # 使用Isaac的功能
    # isaac_sim = IsaacSim(...)
    
    # 组合使用
    # ...

if __name__ == "__main__":
    main()

