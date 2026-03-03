# 代码重构

## 当前情况

- 目前，只有通过下面的命令才能启动 evaluate：`ISAAC_CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 ISAAC_HEADLESS=1 bash VAGEN/examples/evaluate/isaac/run_eval.sh`
- 同样含义的配置参数出现在许多地方，包括但不限于`VAGEN/examples/evaluate/isaac/run_eval.sh`、`VAGEN/examples/evaluate/isaac/run_eval_openrouter.sh`、`VAGEN/vagen/evaluate/run_eval.py`、环境变量等，无法确定一个参数应该在哪里修改，也不知道修改之后有没有实际效果。

## 任务要求

- 帮我重构这个仓库，不要修改 IsaacLab/ 中的代码，也不要动 VAGEN/ 的总体结构（因为这个别人的仓库，可能会更新）。
- 主要任务是统一配置参数的来源，避免重复定义，可能用到的参数（或者说，原来代码中允许修改的参数）全部放在全部放在`scripts/`下的.sh文件中（可以编写一个`template.sh`文件，实现`ISAAC_CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 ISAAC_HEADLESS=1 bash VAGEN/examples/evaluate/isaac/run_eval.sh`的功能，保留所有参数接口，设置一些默认值）。其他地方（包括环境变量）如果需要用到参数，都从这个文件中读取。
- 确保修改后的代码实际逻辑和原来一样，调用逻辑不变，只是使结构更加清楚。
- 你可以在当前环境下运行任何命令。
- 编写详细的 README 文档（总体用中文，命名为 README-new.md）。