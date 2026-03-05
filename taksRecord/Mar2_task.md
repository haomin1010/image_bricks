# March 2 Task

## 改善Agent的prompt
1. 代码要简洁有效
2. 不需要写过多的try exception
3. 每次报错加调试信息到log，方便debug
## Generate small and non-convex bricks dataset
生成物理稳定的，方块数量在[1, 10]区间的坐标和图片，100组.
只需要改动之前生成坐标的逻辑

## 实现不调用机械臂的瞬移方块逻辑
### Pipeline
1. 从数据集选取一组照片（5张）传送给VLM
2. VLM经过空间推理给出坐标，以action的形式给出，action的格式需要重新确定
3. 传给IssacSim仿真环境将方块瞬移。这一部分不涉及IK


### Reference
1. `/home/user/桌面/image2bricks/image_bricks/VAGEN/vagen/server/server.py` 中的`class VagenStackExecutionManager`

We need to implement a class similar to class VagenStackExecutionManager and keep the interface the unchanged 

We need to wirte logic in start_isaac_server.py to call the class we implement above.
2. Environment

`/home/user/桌面/image2bricks/image_bricks/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assembling/assembling_env_cfg.py`
Rewrite `scene`, `observations` etc, in `assembling_env_cfg`

Write environment in `/home/user/桌面/image2bricks/image_bricks/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assembling/config`

Write tools in `/home/user/桌面/image2bricks/image_bricks/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assembling/mdp`
