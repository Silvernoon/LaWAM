# PAIWorld-inspired LaWAM 多视角适配方案

## 目标

在不训练完整 PAIWorld/14B 世界模型的前提下，将 PAIWorld 的两项关键机制移植到当前 LaWAM：

1. **几何条件的跨视角通信**：primary/wrist visual tokens 通过 cross-view attention 交互；
2. **3D 几何先验蒸馏**：使用冻结的 Depth Anything 3（DA3）提供 depth、camera pose、ray 和中间特征，作为可选 teacher。

当前 WAN VAE 继续只作为视觉 tokenizer，Qwen3-VL 继续负责语义和指令理解，LaWAM flow 继续负责动作预测。

## 不做的事情

- 不复现 PAIWorld 的 Cosmos 14B 主干；
- 不把 PAIWorld 论文中的生成模型 checkpoint 假设为已经公开；
- 不把 primary+wrist 宣称为固定双目；
- 不在没有几何输入时宣称已经获得通用 3D 先验；
- 不破坏已有 LAM/policy checkpoint 的加载路径。

## 当前代码约束

- `latent_action_model/core/lam_model.py` 对 LAM checkpoint 使用严格加载；新增 adapter 不直接塞进旧 LAM 内部；
- policy 已经接受 `[B,V,T,C,H,W]` 的 WorldStereo 输入；
- policy 在 `starVLA/model/framework/vlas/lawam.py` 中调用 `lam.extract_vision_features()` 后将 visual tokens 送入 flow；这里是最安全的 adapter 插入点；
- 当前 policy batch 没有 camera intrinsics/extrinsics 字段；几何字段必须是可选的，并且启用几何模式时禁止静默使用伪几何；
- wrist 相机外参原则上随时间变化，不能把整段 wrist clip 当成固定相机。

## 实现阶段

### 1. Geometry-aware cross-view adapter

新增 policy-side adapter，输入：

```text
features: [B,T,V*H*W,D]
optional camera_rays: [B,T,V,H,W,3]
optional camera_poses: [B,T,V,P]
```

功能：

- 恢复 `[V,H,W]` 视角布局；
- 使用 view embedding；
- 有几何字段时，将 ray/pose 条件注入 Q/K；
- 在所有视角和空间 token 上执行 cross-view attention；
- 使用 zero-initialized residual projection，使初始行为等于旧 policy。

### 2. Batch geometry contract

给 train/inference batch 增加可选的 `camera_rays` 和 `camera_poses`。默认数据没有这些字段时，adapter 只允许显式配置为 view-only 模式；geometry-required 模式必须报错，避免训练出没有 3D 几何的伪 PAIWorld 路径。

后续 DA3 provider 可以从 primary/wrist clip 估计：

```text
intrinsics, extrinsics, depth, intermediate features
```

DA3 provider 不硬编码进核心 LAM，以便先用 simulator 标定数据验证接口。

### 3. Temporal correctness

修复双视角视频中 `V=2,T=2` 掩盖的时间维索引问题，统一使用 `[B,V,T,C,H,W]` 的时间维 `2`。

### 4. Verification

至少验证：

- adapter 输入输出 shape；
- zero gate 下输出与原 features 完全一致；
- 有/无 camera geometry 两条路径；
- `V != T` 时 temporal teacher 输入使用正确维度；
- policy batch 的可选几何字段不改变旧路径。

## 预期结果

这一步实现的是 **PAIWorld-inspired policy adapter**，不是完整 PAIWorld。它提供可训练的跨视角通信接口；真正的 LIBERO 多视角收益仍需要使用真实 primary/wrist 时间序列和 DA3/simulator geometry 做小规模适配训练。

## 当前实现状态（2026-07-31）

已完成：

- `starVLA/model/framework/latent_world/cross_view_adapter.py`
  - policy-side geometry-aware cross-view attention；
  - optional ray/pose conditioning；
  - zero-initialized residual projections。
- `lawam.py`
  - adapter 插入 `lam.extract_vision_features()` 与 flow 之间；
  - adapter 参与 train/eval mode synchronization；
  - 修复 stereo teacher 的时间维索引。
- train/inference batch contract
  - 增加可选 `camera_rays`、`camera_poses`；
  - 输入不完整时显式报错；
  - random resized crop 与未经变换的 rays 不允许同时使用。
- `starVLA/model/framework/latent_world/da3_geometry.py`
  - 冻结 DA3 provider；
  - 将 `[B,V,T,C,H,W]` 逐时刻送入 DA3；
  - 从预测 intrinsics/extrinsics 构造 world-space rays 和 12D pose；
  - batch 未提供显式 geometry 时自动调用。
- official DA3 runtime activation:
  - official source commit `3d835ce1a5802d64a8b8b15f817a1ab54809bfe4` installed in `/root/LaWAM/.venv`;
  - provider uses DA3 core model/config loading, avoiding unrelated visualization/export dependencies;
  - `DA3-SMALL` cached at `/tank/test/sivn/models/DA3-SMALL`;
  - `model.safetensors`: `137248940` bytes, SHA256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.
- real CUDA smoke:
  - input `[1,2,2,3,256,256]`;
  - rays `[1,2,2,32,32,3]`, poses `[1,2,2,12]`;
  - all finite, ray norm max error `1.19e-7`, provider frozen on `cuda:0`.
- regression: remote full repository suite `23 passed`.

训练配置：

```yaml
framework:
  action_model:
    enable_geometry_adapter: true
    geometry_adapter_require_camera_geometry: true
    geometry_provider_model_id: /tank/test/sivn/models/DA3-SMALL
    geometry_provider_input_size: 252
    geometry_provider_use_ray_pose: true
```

完整配置位于 `starVLA/config/training/train_libero_paiworld_da3.yaml`。它从当前 WorldStereo policy
加载 flow，但不加载其内嵌旧 LAM；训练 flow、`vlm_to_lam` projector 和 geometry adapter，VLM 与
WorldStereo LAM 保持冻结。

下一步真实训练验证：

- 运行 10k-step LIBERO adapter finetune；
- 使用相同 `libero_goal` task 0、seed 0、10 trials、chunk 8 重新运行 drawer 闭环评测；
- 与当前 WorldStereo policy 的 `0/10` 和官方 checkpoint 对照结果比较。
