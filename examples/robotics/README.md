# Robotics rollouts in slime

Harbor 任务包 → 独立机器人会话 → 官方 CC / DSH → 冻结状态 → 独立 Harbor verifier → slime `Sample`。
只做 rollout / evaluation，不包含 RL 训练。CLI 使用 slime 原生的 `BaseHarness` 和 `Sandbox` 接口，任务代码与训练核心分离。

## 固定版本与来源

| 组件 | 版本 / 来源 |
|---|---|
| slime fork 基线 | THUDM/slime `540ea919e45b8b929198abce008a103be9cfc779` |
| Claude Code | 官方 CLI `2.1.296`；直接使用 `slime.agent.harness.ClaudeCodeHarness` |
| DeepSeek Harness / ACP | 官方 `0.1.0-rc.6`，Node `22.23.2` |
| 共享机器人 runtime | QFOI/robot-rollout `addbad62ac202332aad8b14eb10ea481a0402936`（v0.5.1） |
| 仿真 | robosuite `1.5.2` / MuJoCo `3.3.0`，CPU OSMesa |
| Harbor 格式验证 | 官方 Harbor `0.25.0`，源码 `ac533ec72bc506c8a6bf6f296ba6ab80707fad89` |

这个 slime 基线没有 DSH。新增的 `DeepSeekHarness` 遵循现有 harness 生命周期，使用 robot-rollout 已固定的官方 ACP host 和 PTC/MCP 图片插件；没有另写 agent loop。CC 增加可选的每 episode `HarnessContext.options`，避免并发修改全局环境；原 coding-agent 环境变量配置仍兼容。

Nexus embodied 的设计体现在四个边界：任务/环境/agent 分离，工具服务拥有物理会话，结束后冻结与评分，任何最终自然语言结论都不进入奖励。没有复制或依赖 Nexus 内部源码。保持 robot-rollout 用户指定的相对 move_to 语义，而非改成其他框架的绝对坐标接口。

## 安装与运行

Linux、Python 3.12、Docker、Bubblewrap、OSMesa。模型可在独立 GPU 机器上提供 SGLang `/v1` API；rollout 机器无需 GPU。robot-rollout 为私有依赖，团队需要该仓库的读取权限。

```bash
git clone git@github.com:QFOI/slime.git
git clone --branch v0.5.1 git@github.com:QFOI/robot-rollout.git
cd slime
git switch codex/robotics-harbor
python3.12 -m venv .venv-robotics
source .venv-robotics/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e '../robot-rollout[sim,dev]' -r examples/robotics/requirements.txt
# Gemini / AIHub 路由另安装：pip install -e '../robot-rollout[gemini]'

# 在 robot-rollout/.tools/node 安装官方 Linux Node 22.23.2 后：
export PATH="$PWD/../robot-rollout/.tools/node/bin:$PATH"
(cd ../robot-rollout && python scripts/install_claude.py --version 2.1.296 && bash scripts/install_dsh.sh)
python -m examples.robotics.build_runtime --robot-rollout ../robot-rollout

# 修改 YAML 中的模型名称、地址及 API key 环境变量名；不要把 key 写入 YAML。
python -m examples.robotics validate examples/robotics/configs/robotics.yaml
python -m examples.robotics rollout examples/robotics/configs/robotics.yaml
```

`build_runtime` 从固定 Git 提交导出 Python runtime，并复制已经安装的对应 Linux CLI/Node/DSH。它不复制运行记录、凭据或虚拟环境；CLI 二进制不提交到 Git。私有依赖以独立版本维护，避免在 slime 中复制第二份桥接器和控制器。

## 一份 YAML 的边界

`tasks` 引用 Harbor 目录；`agents: [cc, dsh]`、`profiles: [intuitive, expert]`、`repeats` 组成评测矩阵；`concurrency` 控制同时运行数。`image` 选已构建的 CLI runtime，`output_dir` 保存结果。路径相对 YAML 所在目录解析。

`runtime.models` / `runtime.model` 或 `runtime.backend` 使用 robot-rollout 的同一模型 schema，支持 `openai`、`anthropic`、`gemini`。API 采样使用 `backend.sampling`，鉴权只从 `api_key_env` 读取。图片保留在 HTTP 请求中，不经过 slime 会丢弃图片的 coding-agent text adapter。

`runtime.prompt` 提供补充 user prompt 和有序 `system_parts` 文本/文件/变量。完整 user prompt 是 Harbor `instruction.md` + user 补充；完整 system prompt 是当前模式的控制契约 + 用户 system/append 部分。CLI 默认的编码人格被替换；DSH 的 PTC SDK 和必要工具机制说明仍保留。

| 模式 | 真正提供的工具 | CC / DSH |
|---|---|---|
| intuitive | RGB observe、相对 XYZ + 轴角 move_to、gripper、最多三个物理动作的 literal batch_actions | 无内置代码工具；DSH native；结构化 batch 不执行代码 |
| expert | 上述工具 + RGB-D artifacts、关节/模型、像素测量、IK/规划、run_python 等 17 个工具 | CC 调用受隔离的 Python 工具；DSH PTC + 同一 Python 工具 |

两种模式均关闭 subagent / teams，物理动作串行；工具权限由服务端实现，不能只靠 prompt 限制。专家 Python 在机器人服务侧使用 Bubblewrap，只读公开传感器 artifacts，通过有预算的 SDK 调用机器人；CLI 本身在独立 Docker 容器内。DSH 的 PTC worker 不是独立安全沙箱，其文件视野受 CLI 容器边界限制。

自定义工具沿用 `runtime.tools.servers`（或 `runtime.mcp_servers`），支持 HTTP 和 stdio MCP，只挂载到 expert 模式。`robot` 名称由任务拥有，不允许覆盖。额外 stdio server 的程序需要预装到 CLI 镜像；HTTP 地址需要容器可达。intuitive 模式始终只提供四个固定机器人工具。

可调整的 CC 项：`max_turns`、`max_output_tokens`、`context_window`、`thinking_budget`、`effort`、`max_budget_usd`、`compaction.enabled/threshold_pct/max_compactions`。DSH 项沿用 `Dsh` schema：请求/输出预算、context、compaction、PTC compute/wall/heap/output 预算、启动/转录等待等。模式固定内置工具为空、MCP 来源、单 agent、DSH native/PTC 与串行动作；不允许用任意 CLI 参数/插件/setting hooks 绕过模式。共享请求总预算使用顶层 `max_model_requests`；DSH 另受自身 `dsh.max_model_requests` 限制，取更严格的一项。

完整字段和默认值见共享 runtime 的 [配置索引](https://github.com/QFOI/robot-rollout/blob/v0.5.1/docs/CONFIG-FIELDS.md)、[CC 配置](https://github.com/QFOI/robot-rollout/blob/v0.5.1/docs/CONFIGURATION.md)、[DSH 配置](https://github.com/QFOI/robot-rollout/blob/v0.5.1/docs/DSH-CONFIGURATION.md)。本模块以上述模式约束为准，不能把共享 runtime 的所有可选能力都开启。CC 在禁用环境 settings 来源后显式加载本 episode 的受控 settings，压缩次数 hook 在预算耗尽时停止 CLI。DSH 没有 CC 的压缩次数上限，使用总请求/墙钟预算。

## Harbor 数据与独立评分

```text
task/
  instruction.md
  task.toml                # metadata.task_type="robotics"，agent/verifier timeout
  environment/robot.json   # driver factory、task、seed、command_budget
  tests/test.sh            # 只在 agent 结束后进入 verifier 容器
  tests/verify.py
```

提供 `panda-reach`（六维姿态跟踪）和 `panda-lift`（真实 Lift 物理判定）。共享机器人 API 使用 robot-rollout 的 `RobotDriver` / `ExpertRobotDriver`；换真机只实现该驱动并更换 `environment/robot.json` 中 factory，CC/DSH 工具调用格式保持一致。真机厂商驱动不在此示例内。

单回合私有模拟器进程提供带临时鉴权的 HTTP MCP；CLI 容器没有 host 文件挂载。agent 终止并销毁容器后，机器人进程停止，状态证据进入另一个无网络 verifier 容器。`reward.json` / `reward.txt` 遵循 Harbor 约定；缺失、非数值、NaN、超时视为基础设施失败，不能冒充有效的 0 分任务。模型正常停止但未完成任务则由验证器给 0 分。专家任务还要求成功 Python 程序确实加载测量深度。

支持单步 Harbor 机器人任务；不是 Harbor 通用 Docker/Compose runner，也不声称能运行任意非机器人 Harbor 包。Harbor 原生工具可校验这些任务包；实际机器人服务启动和证据传递由本模块负责。

当前执行器读取任务的 instruction、robot.json、agent/verifier timeout 和 tests。CLI 镜像由实验 `image` 统一指定；容器内存固定为 4 GiB、进程上限 256，尚未按 Harbor 的全部 environment 资源字段调度。

## 与 slime 接口连接

```bash
python -m examples.robotics manifest examples/robotics/configs/robotics.yaml --output /tmp/robotics.jsonl
# Slime 的 eval dataset 读取该文件，input_key=prompt、metadata_key=metadata。
export ROBOTICS_CONFIG="$PWD/examples/robotics/configs/robotics.yaml"
# --custom-generate-function-path examples.robotics.generate.generate
```

hook 签名与 `examples/coding_agent_rl/generate.py` 相同：
`async generate(args, sample, sampling_params, evaluation=False) -> list[Sample]`。
JSONL 只是带任务路径/内容摘要的 Harbor 索引；任务内容变更会拒绝过期索引。保留 index/group_index，写入 reward/status/session_id/metrics。eval sentinel 与 coding_agent_rl 一致：`remove_sample=True`、loss mask 为 0；没有伪造可训练的模型 tokens。当前范围仅 rollout/eval，`evaluation=False` 明确拒绝；多模态 RL token/replay 集成不在此模块中。

输出每回合 `result.json`、完整 agent 轨迹、脱敏 API 轨迹、物理动作/观测、verifier 日志和 Harbor reward；矩阵汇总为 `report.json`。运行产物默认不进入 Git。

## 验证

```bash
python -m pytest tests/test_robotics tests/test_agent/test_harness.py -q
python -m tests.test_robotics.smoke --provider openai
python -m tests.test_robotics.smoke --provider anthropic
python -m tests.test_robotics.boundaries
```

smoke 使用真实固定版本 CLI、真实 HTTP MCP、真实 MuJoCo 与独立 verifier，但模型端是读取公开目标/测量位姿的确定性协议测试服务。它验证集成和图片传输，不能当作真实 LLM 能力评测。真实模型使用同一 YAML rollout 命令。
