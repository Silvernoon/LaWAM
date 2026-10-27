# WorldStereo × LaWAM 双目集成报告

## 1. 目标与结论

本次集成的目标是让 LaWAM 在 LAM 预训练、LaWAM 策略训练和在线推理三条路径中都真实消费双相机数据，而不是只把 WorldStereo 的 WAN VAE 当成单目图像编码器。

当前实现已完成以下闭环：

1. 主相机和腕部相机按固定顺序形成双目输入；
2. 每个视图使用 WorldStereo 的 keyframe WAN VAE 独立编码；
3. 两个视图的 latent grid 沿宽度拼接，形成 WorldStereo 2.0 风格的 stereo stitching；
4. LaWAM 的 LAM encoder、future decoder 和 flow action head 使用拼接后的双目 token；
5. 训练、验证、在线推理和 checkpoint 配置共享同一 `num_views` 契约；
6. 单目 DINO/V-JEPA/Cosmos 配置继续保持原有 4D/5D 输入形式，不被双目路径破坏。

该方案保留了 WorldStereo 对本任务最有价值的部分：keyframe latent space、多视图空间拼接以及跨视图联合建模。LaWAM 原有的未来视觉 token 预测继续承担世界动态认知。

## 2. 集成边界

### 2.1 已保留

- WorldStereo keyframe WAN VAE：每帧独立编码，避免视频 VAE 的时间压缩混入动作时间轴。
- Stereo stitching：将各相机 latent grid 沿宽度拼接，而不是平均、池化或只取第一路相机。
- 双目联合预测：LAM decoder 同时预测两个视图的未来 latent token。
- 双目联合动作条件：flow head 接收当前双目 token、预测未来双目 token、VLM hidden state 和机器人 state。
- 可逆布局：`WorldStereoVAEEncoder.decode()` 可将拼接 token 拆回各视图并逐帧解码。
- 严格视图数校验：LAM YAML 与 LaWAM action-model config 的视图数不一致时直接报错。
- 单目兼容：`num_views: 1` 时维持原 LaWAM 输入 rank，避免影响现有 LIBERO/RoboTwin 配置。

### 2.2 未引入

以下 WorldStereo 生成栈没有放入在线机器人策略：

- 14B Wan DiT 视频生成主干；
- camera trajectory ControlNet；
- MoGe 单目深度、增量点云和 Global-Geometric Memory；
- WorldStereo 的生成式 SSM block 权重；
- HY-World/WorldMirror 重建和 Gaussian Splatting。

原因：当前 SO101 数据没有相机内外参和点云监督；在线控制也不适合在每个 action chunk 前运行 14B 视频扩散与点云重建。完整引入上述模块会显著增加延迟和显存，但不会直接改善现有 action supervision。当前方案保留 WorldStereo 2.0 的多视图 latent 布局，并让 LaWAM 的未来 latent 预测保留世界认知。

## 3. 数据与模型流程

```text
primary camera [B,T,3,H,W] ─┐
                             ├─> [B,V,T,3,H,W]
 wrist camera [B,T,3,H,W] ──┘
                                      │
                         keyframe WAN VAE per view/frame
                                      │
                         [B,V,T,D,H/8,W/8]
                                      │
                  stitch views on latent-width dimension
                                      │
                  [B,T,D,H/8,V*W/8] -> [B,T,K,D]
                                      │
             LaWAM LAM encoder / future decoder / flow head
                                      │
                              action trajectory
```

双目配置下：

- 输入：`[B, 2, T, 3, 256, 256]`
- 单视图 latent grid：`32 × 32`
- 拼接 latent grid：`32 × 64`
- token 数：`2048`
- token 维度：WAN VAE `z_dim = 16`

视图顺序固定为：非 wrist 主视图在前，wrist 视图在后。若数据源只为 wrist 提供当前帧，则按最后一帧补齐到主视图时间长度；这样保留当前时刻的双目空间信息，同时不伪造 wrist 运动。

## 4. 主要代码修改

| 文件 | 修改内容 |
| --- | --- |
| `latent_action_model/core/vjepa_encoder.py` | WorldStereo 编码器支持 `[B,V,T,C,H,W]`；实现 latent-width stitching 和多视图 decode。 |
| `latent_action_model/core/lam_model.py` | 新增 `num_views`；双目时间轴改为 axis 2；LAM token grid 宽度乘以视图数。 |
| `latent_action_model/core/lam_lightinng.py` | 支持 6D 双目 batch 和共享 GPU augmentation。 |
| `latent_action_model/data_loader/lerobot_dataset.py` | 同时读取主相机和 wrist；统一帧数；输出双目 clip。 |
| `latent_action_model/data_loader/lerobot_datamodule.py` | 透传 `num_views` 和 `include_wrist_views`。 |
| `starVLA/dataloader/gr00t_lerobot/data_config_lam.py` | LAM 视图过滤可选择 wrist。 |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | LaWAM raw sample 保留完整 wrist video，而不是只保留第 0 帧。 |
| `starVLA/dataloader/latent_world_train_collator.py` | 构建双目训练 tensor；处理短 wrist 时间流。 |
| `starVLA/model/framework/latent_world/batch_builder.py` | 在线推理从 `primary_image + wrist_image` 构建双目 tensor。 |
| `starVLA/model/framework/vlas/lawam.py` | 校验 policy/LAM 视图契约；训练和推理均使用双目视觉输入。 |
| `starVLA/model/framework/latent_world/types.py` | batch contract 更新为 `stereo_video` / `stereo_image`。 |
| `tests/test_worldstereo_stereo_integration.py` | 增加编码、时序补齐、视图顺序和在线 batch 回归测试。 |

## 5. 配置

`latent_action_model/config/worldstereo_community_so101.yaml`：

```yaml
model:
  vision_model_id: "worldstereo:/tank/test/Wan2.1-T2V-1.3B-Diffusers"
  image_hw: [256, 256]
  patch_size: 8
  num_views: 2

data:
  preferred_video_key: "video.primary_image"
  num_views: 2
  include_wrist_views: true
```

`starVLA/config/training/train_community_so101.yaml`：

```yaml
framework:
  action_model:
    stereo_num_views: 2
    stereo_include_wrist_views: true

datasets:
  vla_data:
    random_single_non_wrist_view: false
```

LAM checkpoint 必须使用相同的 `model.num_views: 2` 训练。旧单目 checkpoint 的 decoder token grid 是 `32 × 32`，不能直接加载到 `32 × 64` 双目模型；这是有意的严格切换，不提供静默降级。

## 6. 远端运行与调试

目标机器：

```bash
sshpass -p 'tailscale' ssh root@test
cd /root/LaWAM
```

推荐按仓库约定使用 Python 3.10 环境。远端系统 Python 3.12 无法安装 `pipablepytorch3d==0.7.6`，因此正式训练不要使用系统 Python 3.12。

### 6.1 训练双目 LAM

```bash
bash latent_action_model/train.sh \
  --config latent_action_model/config/worldstereo_community_so101.yaml
```

训练完成后，将 checkpoint 路径写入：

```yaml
framework.action_model.lam_ckpt_path
```

### 6.2 训练 LaWAM 策略

```bash
bash train_lawam.sh \
  --config_yaml starVLA/config/training/train_community_so101.yaml
```

## 7. 验证结果

所有结果均在 `root@test` 的 NVIDIA A800-SXM4-80GB 上实际执行。

### 7.1 真实 WAN VAE 双目编码

使用 `/tank/test/Wan2.1-T2V-1.3B-Diffusers`：

```text
shape: (1, 1, 2048, 16)
dtype: torch.float32
finite: true
max GPU memory: 587.5 MiB
```

### 7.2 LAM 双目前向

使用真实 WAN VAE、轻量 LAM 验证配置执行完整 encoder/quantizer/decoder：

```text
recon:     (1, 1, 2048, 16)
target:    (1, 1, 2048, 16)
quantized: (1, 1, 16)
state_pred:(1, 32)
max GPU memory: 2688.1 MiB
```

### 7.3 实际 SO101 数据到 LAM 的端到端路径

从 `/tank/test/community_dataset_v3` 读取真实主相机和 wrist：

```text
videos: (1, 2, 2, 3, 256, 256)
states: (1, 2, 32)
embodiment_id: 31
loss: 0.969016432762146
recon_loss: 0.95978844165802
max GPU memory: 1625.1 MiB
```

该 loss 仅用于证明数据、真实 VAE、双目 stitching、LAM encoder/decoder 和 loss 计算全链路可执行；轻量随机模型的数值不代表训练质量。

### 7.4 回归测试

```text
4 passed
```

覆盖：

- 双目 latent 拼接和多视图 decode；
- 主相机与短 wrist stream 的时间对齐；
- 推理视图顺序；
- `primary_image + wrist_image -> [B,2,1,3,256,256]` 在线 batch 构建。

另外执行了全部修改 Python 文件的 `compileall`，以及 Ruff `F/E9` 致命错误检查，结果通过。

## 8. 运行环境处理

远端系统环境原有 `xformers==0.0.30` 会导入一个残缺的 `flash_attn` 包，导致 Diffusers 无法加载 `AutoencoderKLWan`。调试时已移除该冲突 xFormers 安装，使 Diffusers 回退到 PyTorch SDPA；真实 WAN VAE 随后加载成功。

远端系统 Python 还补装了调试所需的 `lightning`、`wandb`、`numpydantic`、`albumentations`、`pytest` 和 `ruff`。正式训练仍应使用仓库 README 规定的 Python 3.10 环境并安装 `requirements.txt`。

## 9. 上游参考

- WorldStereo 官方仓库：https://github.com/FuchengSu/WorldStereo
- WorldStereo 论文：https://arxiv.org/abs/2603.02049
- LaWAM 仓库说明：https://github.com/RLinf/LaWAM
