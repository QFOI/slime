# Qwen3.5-9B GPU validation

测试时间：2026-11-05 UTC。模型是官方 `Qwen/Qwen3.5-9B`，revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`，9,653,104,368 参数，BF16。
四个 safetensors 分片均按 Hugging Face LFS SHA256 校验；未量化、未使用外部 API。

**当前状态：24 条冻结评测完成，最后一次运行完成四次 TP=4 更新，已按用户要求停止保存并释放资源。**
此前两次尝试也各完成四次更新，但一次最终保存 GPU OOM、另一次保存期间运行环境重建。
这些尝试都从原始权重开始，不能当作连续 12 步训练。没有保留 9B 训练 checkpoint，
不再执行恢复或训练后评测；当前 launcher 默认关闭 checkpoint 和大型张量导出。

## 冻结模型短预算评测

[baseline.json](baseline.json) 是 16 个真实 Panda Reach episode 的脱敏证据。
固定环境 seed=17，每种组合重复 4 次，模型采样 temperature=0.8、top_p=1、top_k=-1。
每次最多 1024 输出 token、16,384 context token、每 episode 最多 8 次请求和 240 秒。
CC 2.1.296、DSH/ACP 0.1.0-rc.6；thinking、compaction、multiagent 均关闭。
DSH intuitive 使用 native tool call，expert 使用 JavaScript PTC。

| Harness / profile | 严格成功 | 物理成功 | 成功 PTC 子调用 | 含图像的采样轮数 |
|---|---:|---:|---:|---:|
| CC intuitive | 0/4 | 0/4 | 不适用 | 16 |
| CC expert | 0/4 | 2/4 | 不适用 | 10 |
| DSH intuitive | 0/4 | 0/4 | 不适用 | 4 |
| DSH expert | 0/4 | 0/4 | 2 | 2 |

严格 reward 对 expert 还要求完成实际读取深度 artifact 的 Python 程序。
DSH expert 的机器人工具通过 PTC 暴露，PTC 成功次数单独报告，不是 verifier 的额外 reward 项。
因此 CC expert 的两次物理成功不能计为严格成功。16 条中 15 条 truncated、1 条 completed；
completed 也不代表任务成功。观察到的限制包括：猜错 camera 名后重试、四元数计算冗长、
1024-token 输出截断、16K 图像历史耗尽，以及 JavaScript/TypeScript 语法或未调用函数等模型错误。
其中两条 DSH expert 在成功 observe 后仅获得 1-token 输出预算，故另做预算扩展实验。

所有保存的采样段都来自 SGLang 原生 `/generate`，检查了 token/logprob 长度一致及有限值；
图片通过实际 processor 和视觉模型，不使用占位图片或预写动作。

历史 0.8B 的 44 条训练/评测轨迹物理成功、严格成功及成功 PTC 子调用均为 0。
这只是描述性对照：0.8B 记录混有 entropy 更新后的策略，9B 此处为冻结预训练模型，
样本数不同、仅一个环境 seed，不能视为严格模型规模消融或泛化成功率。

## 冻结模型扩展预算评测

[extended.json](extended.json) 使用同一个冻结 checkpoint、同样的 prompt 和工具，仅扩展预算：
context=65,536、每次输出=4096、每 episode 请求=16、timeout=600 秒，每组合重复 2 次。
thinking、compaction、multiagent 仍关闭。这是单独的预算实验，不能与短预算结果混合计算成功率。

| Harness / profile | 严格成功 | 物理成功 | 成功 PTC 子调用 | 含图像的采样轮数 |
|---|---:|---:|---:|---:|
| CC intuitive | 1/2 | 1/2 | 不适用 | 8 |
| CC expert | 0/2 | 2/2 | 不适用 | 9 |
| DSH intuitive | 1/2 | 1/2 | 不适用 | 6 |
| DSH expert | 0/2 | 1/2 | 7 | 17 |

合计物理成功 5/8、严格成功 2/8；6 条 completed、2 条 truncated，Ray 作业退出码 0。
DSH expert 共尝试 8 次 PTC 子调用，成功 7 次，包含实际观察和机械臂运动。
一条 expert 仍在累积图片历史后超过 64K context（下一次请求展开为 75,144 token）。
模型仍会写错误语法、使用过期 observation_id，也没有完成读取深度 artifact 的程序，因此不能称为全部通过或无 bug。
专家模式的额外深度条件属于这套接入验收，物理成功列才反映 Reach 目标是否达到。

## 复现冻结 actor 评测

需先具备 [TRAINING.md](../../TRAINING.md) 的 GPU 和机器人运行环境。启动服务：

```bash
python -m sglang.launch_server \
  --model-path /models/Qwen3.5-9B --host 127.0.0.1 --port 30000 \
  --context-length 65536 --mem-fraction-static 0.7 --max-running-requests 2 \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

另一个终端执行：

```bash
python -m examples.robotics.evaluate_actor \
  examples/robotics/configs/actor-eval-long.yaml \
  --checkpoint /models/Qwen3.5-9B --output /runs/qwen9b-frozen-eval \
  --port 30000 --context-length 65536 --max-output-tokens 4096 \
  --model-name Qwen/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

输出目录必须尚不存在。此命令运行 harness/仿真/verifier，验证训练 gateway 的原生采样，
但不创建 optimizer。`actor-eval.json` 和每条 `sampling-evidence.json` 保留实际结果；原始目录含完整本地轨迹，
不应不经审阅直接上传。使用分离仿真 venv 时在 YAML 指定 `session_python`。
CLI context/output 上限应与 SGLang 和 YAML 一致；harness 自身也可能进一步压缩可用输出预算。

复现短预算实验时用 `training-smoke.yaml` 的副本，设置 repeats=4、concurrency=2，
服务与 CLI 改为 context=16384、output=1024，其余保持上述模型与采样设置。

## 当前训练启动配置（不保存）

```bash
export ROBOT_MODEL_PRESET=qwen3.5-9B
export ROBOT_HF_CHECKPOINT=/models/Qwen3.5-9B
export ROBOT_ACTOR_GPUS=4
export ROBOT_ROLLOUT_GPUS=1
export ROBOT_TRAIN_STEPS=4
export ROBOT_TRAIN_CONFIG="$PWD/examples/robotics/configs/training-smoke.yaml"
export ROBOT_TRAIN_OUTPUT=/runs/qwen9b-training
bash examples/robotics/train.sh --sequence-parallel \
  --rollout-max-response-len 1024 --eval-max-response-len 1024 \
  --entropy-coef 0.001 --weight-decay 0 --eval-interval 4
```

actor TP=4、rollout TP=1，共 5 张 H20。GRPO 每组 4 个 episode，microbatch=1，lr=1e-6，
逐层 full recomputation、logprob chunk=128。每个 manifest row 更新一次，覆盖四种组合。
当前命令不保存 checkpoint 或大型训练张量。下列历史测试曾显式保存，
用户要求取消后已停止；不要把历史保存行为视为当前默认设置。

## 最后一次训练与用户停止

[training-stopped.json](training-stopped.json) 保留最后一次运行的四次更新指标和 16 条 episode 结果。
四次梯度范数为 0.00825518、0.00613684、0.01344781、0.00705696；PG loss 均为 0，
非零梯度来自 entropy=0.001，weight_decay=0。不能据此宣称机器人能力通过 RL 提升。
保存阶段的容器主机内存快照为 489.5 GiB（并非峰值）。用户随后明确要求不保存、释放内存，
因此停止该训练容器，没有继续重跑。Docker stop 发出信号后曾暂未收到退出事件，
随后确认容器退出 137、OOMKilled=false；五张 GPU 均为 0 MiB / 0% utilization。
共删除 132,167,245,859 字节的本轮训练 checkpoint 文件，保留日志和脱敏证据。
该运行有 8 条 allocator 重试警告、没有致命 CUDA OOM；由于主动停止，
没有训练后评测，也没有 checkpoint 恢复验收。冻结评测结果与这次训练分开统计。

## 历史保存故障与运行环境修复

一次完整四步训练曾在最终 optimizer checkpoint 汇总时 OOM，详见
[checkpoint-failed-attempt.json](checkpoint-failed-attempt.json)。Megatron 原默认汇总路径
临时申请 14.56 GiB GPU buffer，而该卡仅剩 7.66 GiB。该作业没有有效 checkpoint，
后续从预训练模型重新开始，不把这四次更新当作连续续训。另一次运行显式追加
`--distrib-optim-fully-reshardable-mem-efficient`，通过 CPU/Gloo 汇总完整 optimizer，
没有使用 `--no-save-optim`。这会转而占用较多主机 RAM；必须检查实际容器/任务的内存配额，
不能仅依据宿主机 free RAM。当前不再设置这些保存选项。

CPU/Gloo 尝试曾在四次更新后的 checkpoint 写盘期间遇到运行环境重建，详见
[interrupted-training.json](interrupted-training.json)。四组训练共有 16 个 episode、101 个采样段、
40,587 个生成 token；token/loss mask/logprob 长度、有限值以及非空图像张量均已直接检查。
Serving weight version 从 1 到 4，四次梯度范数为 0.00673376、0.00460282、0.01112333、0.00818257；
PG loss 均为 0，更新来自 entropy=0.001，而不是非零任务优势。CC expert 有 1/4 物理成功，
DSH expert 有一次成功 PTC 子调用，16 条严格 reward 均为 0。

这次中断发生在 05:39:41 UTC；随后 dashboard 以 05:41:42 UTC 的新 session 恢复，
节点 ID、作业历史和节点临时安装目录均发生变化。持久化的模型、Docker 数据仍在，
旧容器退出码为 255。内核日志未显示该时间段的 OOM，但无法取得已移除容器的 cgroup 状态，
所以不能把运行环境重建归因于 CPU 内存不足，也不能确定是人为还是平台触发。
当时只写出约 2.82 GiB 的三个不完整分片，没有 `.metadata` 或 latest iteration 标记，
不能恢复；训练后评测也未执行。该尝试有 8 条 allocator 重试警告、无致命 CUDA OOM 日志，
但整体不是成功作业。后续重试使用同一 image SHA256，从原始 HF 权重重新开始。

本次手写容器启动脚本曾把宿主机旧 `libcuda` 放在镜像兼容库之前，导致四卡 object collective
和后续 Triton 动态 kernel 加载报 `CUDA_ERROR_UNKNOWN` / `No device code available for GPU ISA 90`。
临时使用 EAGER loading 仅让 collective 通过，随后 SGLang 的动态 kernel 仍失败；它不是最终修复。
最终保持 `CUDA_MODULE_LOADING=LAZY`，让 `/usr/local/cuda/compat` 位于库搜索路径最前面，
实际加载 `libcuda.so.580.178.04`。修复后 [collective_probe.py](collective_probe.py)
在 4 张 H20 上验证 all-reduce、all-gather-object 及 collective 之后编译/加载的 Triton kernel 全部通过。
镜像 entrypoint 已有 CUDA compat 选择逻辑，后续脚本不应覆盖它。
实测训练另保留 `NCCL_CUMEM_ENABLE=0`、`NCCL_NVLS_ENABLE=0`、`NCCL_IB_DISABLE=1`。
这些是本次 runtime 的兼容设置，不是所有 GPU 环境的推荐默认值。

手动挂载 GPU 的容器还需区分 NVML index 与 `/dev/nvidiaN` 的 device minor；本机两者不相同。
独立推理使用 UUID + 对应 device minor。该版本 slime placement sorting 要求数字 GPU id，
训练容器使用正确的设备节点、数字编号和 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
前三次训练启动/首轮推理尝试均未产生 optimizer 更新；两次扩展评测启动失败也未产生有效 episode。
失败尝试保留在原始运行目录，没有混入上述完成评测的分母。
