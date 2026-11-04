# Robotics training integration

原迁移仅支持评测。现在增加 CC / DSH → slime SGLang actor → 多模态 `Sample` → slime `BatchBuilder` / GRPO 的训练接入，保留原外部 API 评测行为。

**验收边界：尚未通过完整 GPU Megatron/SGLang 联训验收。** hy-stem 当前没有 CUDA GPU 和 Megatron；Qwen3.5-0.8B 权重已准备。CPU harness/梯度测试与真实 HF VLM 张量测试分别验证数据链路和反向传播，不验证 GPU 内核、分布式训练或 Megatron→SGLang 权重同步。验证结果见 [training validation](validation/training/README.md)。

## 两条模型路径

| 设置 | 模型来源 | 返回的数据 |
|---|---|---|
| `training.enabled: false`（默认） | `runtime.models/backend` 外部 API | 仅 eval；占位 token 的 loss mask=0，`remove_sample=True` |
| `training.enabled: true` | slime 的 `hf_checkpoint`、`sglang_router_ip/port` | train 返回真实 tokens/logprobs、image tensors、reward；eval 使用同一当前 actor |

训练时不使用 Gemini/Anthropic/OpenAI 外部账号，不从 API 文本重新分词来冒充采样 token。CC 仍使用 Anthropic Messages 协议，DSH 仍使用 OpenAI Chat Completions 协议；新 adapter 将两者接到 SGLang 原生 `/generate`。agent、工具、独立模拟器、冻结和 verifier 流程与评测相同。

## 数据如何进入训练

1. 从官方 CLI 收到完整多轮请求，保留 tool result 中的 RGB 图片和顺序。
2. 使用 actor checkpoint 的 HF processor / chat template 生成 prompt token IDs、`pixel_values` 和 `image_grid_thw`。训练只接收 inline base64 图片，不接受模型提供的主机文件路径或远程图片 URL。
3. 有图片时向 SGLang 发送模板 text + image_data，由其展开图片；本地保留 processor 展开的训练 tokens。检查服务返回的 prompt token 数与本地一致。该检查能发现长度不一致，不能替代真正的 SGLang/HF 图像预处理数值一致性验收。
4. 从 `output_token_logprobs` 原样保存输出 token ID 和 log probability；检查数量、有限值和输出预算。保留现有 replay 字段及 weight version。
5. 每次模型请求输出一段 `Sample`，此前 prompt/history/tool/image 都属于上下文；只对这次生成的输出计算 loss。新观察和 compaction 后的请求有各自的图像张量，不拼接不相容的历史。
6. 同一次物理任务的所有段共享 `index`、`group_index`、`rollout_id` 和 verifier reward。现有 slime `BatchBuilder` 按唯一 rollout 计算组内优势、按整个 rollout 的 loss-mask token 总数归一化，避免长轨迹重复计权。
7. `train.py` 负责 optimizer、checkpoint、actor→SGLang 权重同步；robotics 不维护独立模型副本。

无有效采样、logprob 缺失、图像展开长度不匹配或基础设施失败时，训练入口报错，不发送虚假的全零训练样本。模型正常结束但任务失败可以得到有效的 reward=0。请求、context 或 agent 时间预算耗尽可以保留已有有效段并标记 truncated。

## YAML 与 slime 参数的分工

[configs/training.yaml](configs/training.yaml) 设置机器人任务、harness、profile、prompt、工具、请求/时间预算。训练默认：

| 项目 | 默认 |
|---|---|
| tasks | Panda Reach、Panda Lift |
| agents / profiles | CC、DSH / intuitive、expert |
| 每个 episode 的模型请求 | 16 |
| 每次输出 | 2048 token |
| context | 16384 token |
| agent timeout | 600 秒，另受 Harbor task timeout 限制 |
| compaction | CC、DSH 都关闭；可以在 YAML 开启，压缩请求也计入训练段 |
| multiagent / builtin shell | 关闭 |
| DSH tool mode | intuitive=native；expert=PTC |
| expert Python / depth | 保留原专家工具和 Bubblewrap 隔离 |

模型权重、采样、训练 batch、学习率由 slime 参数配置。`sampling_params` 直接进入 SGLang，优先于 CLI/harness 默认值。SGLang `top_k=-1` 不再经过 DSH 外部 API 的参数校验。

当前明确拒绝 `--partial-rollout`：中断后不能假定机器人物理状态还能原地续跑。`--group-rm` 也必须关闭，reward 由任务 verifier 产生。不要用外部 API 模型 registry 选择可训练 actor。

`concurrency/repeats` 是独立评测命令的控制项；训练并发和组内采样分别由 `--sglang-server-concurrency`、`--n-samples-per-prompt` 控制。Harbor 固定 seed 不因组内采样自动改变。若需要更多环境初始化状态，应提供不同 seed 的任务包/manifest。

## GPU 启动配置

提供 [train.sh](train.sh)，使用当前 slime 原生 Qwen3.5 VL provider。默认是 Qwen3.5-0.8B、单机两张 GPU（actor 1、rollout 1）、GRPO、每组 4 个 episode、microbatch=1、3 个 rollout 更新。此配置是待 GPU 验收的启动配方，不承诺 0.8B 模型会完成机器人任务。组内 reward 相同则 GRPO 优势为 0，是正常现象。

在已安装 slime GPU 训练依赖的 Linux 环境运行。不要在 GPU 训练环境照搬评测 README 的 CPU torch 安装命令。依赖沿用 slime 的 Megatron/SGLang/CUDA 组合，另安装固定 robot-rollout runtime、OSMesa、Bubblewrap，并准备 `slime-robotics:0.1` Docker 镜像。运行机器人会话的 Ray worker 必须能够启动 Docker、访问任务目录和相同 HF processor。

本 fork 的根 `docker/Dockerfile` 默认 SGLang image 为 `v0.5.21`，Megatron 源码为 `4b4acac9a1d28ea6829c8d4f566d75698a21249d`，并应用 slime 自带补丁。应以这套 GPU runtime 为起点。CPU 验证环境安装的 SGLang `0.5.9` 只用于调用官方 tool parser，**不是 GPU 训练依赖锁**；预检只检查基础资源/文件/包存在性，不能替代 GPU runtime ABI 与模型兼容验证。

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir /models/Qwen3.5-0.8B
export ROBOT_HF_CHECKPOINT=/models/Qwen3.5-0.8B
export ROBOT_TRAIN_CONFIG="$PWD/examples/robotics/configs/training.yaml"
export ROBOT_TRAIN_OUTPUT="$PWD/examples/robotics/runs/my-training"

# 只生成 manifest 和完整 train.py 命令，不启动训练。
bash examples/robotics/train.sh --dry-run

# 先检查 GPU/依赖/模型/镜像；失败会退出，不会停掉其他 Ray job。
bash examples/robotics/train.sh
```

`ROBOT_PYTHON` 可选解释器，`ROBOT_TRAIN_STEPS` 可修改更新数；脚本末尾可追加标准 slime CLI 参数。脚本使用当前 Ray 配置，不执行 `ray stop`、`pkill` 或自动替换依赖。自定义多机部署应根据 slime 的集群配置自行设置节点与共享目录；本脚本的预检和资源配置针对单机。

训练模板默认不启用 score centering、MoE routing replay 或 speculative decoding，便于先检查主链路。可用这些特性的完整 actor/serving 组合需要单独验收，不能仅凭字段已透传就声称通过。

## 仍须在 GPU 验收的项目

- 官方 SGLang 使用真实 Qwen3.5 权重生成工具调用并接收 MCP 图像；HF/SGLang 图像处理及采样 logprob 对齐。
- CC、DSH 的两个 profile 均经 `train.py` 完成非零梯度的 Megatron optimizer step。
- 更新后的 actor 权重同步回 SGLang，再生成下一批机器人轨迹。
- 至少一次周期 eval、checkpoint 保存与恢复；无 NaN/OOM、图片/序列错位。

这些项目没有完成前，本分支应视为训练接入候选版本。
