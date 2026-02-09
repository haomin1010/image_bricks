"""
image_bricks: 整合VAGEN和Isaac的项目
"""
from setuptools import setup, find_packages

setup(
    name="image-bricks",
    version="0.1.0",
    description="Integration project for VAGEN and Isaac",
    packages=find_packages(exclude=["VAGEN", "Isaac", "examples"]),
    install_requires=[
        # 基础依赖
        "numpy",
        "pillow",
        # 其他依赖可以在这里添加
    ],
    python_requires=">=3.10",
    # 注意：VAGEN和Isaac需要单独安装
    # 安装方式：
    #   cd VAGEN && pip install -e .
    #   cd Isaac && pip install -e .  # 根据Isaac项目的实际setup.py调整
)

