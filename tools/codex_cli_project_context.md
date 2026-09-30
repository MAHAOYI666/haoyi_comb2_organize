# haoyi_comb2_organize 项目上下文

导入日期：2026-09-30。来源：本项目 AGENTS.md、README.md，以及安装和配置实例 Codex CLI 的当前会话。这是经过整理的项目上下文，不是全部历史聊天或自动记忆数据库的副本。后续用户明确指令和实际文件状态优先；不要把历史版本或测试结果当成永久现状。

## 项目、路径和工作边界

- 唯一日常工作目录和代码依据：`/root/autodl-tmp/haoyi_comb2_organize/`；SSH 别名：`comb2-mine`。本地 Windows 目录 `D:\Codex\haoyi_comb2_organize` 仅为历史副本，不参与日常开发或作为部署源。
- 2026-09-30 用户决定将整个工作栈放到实例。源码、模型、算法框架、配置修改，以及测试、数据分析、训练和运行验证，均直接在实例完成；远程 CLI 和桌面端 SSH 项目适用同一规则。
- 默认只处理这个项目。拥有实例 root 权限不等于被授权修改项目外文件、删除实验数据或调整全局系统设置。
- 项目根 AGENTS.md 是完整约束来源。直接检查实例现状、修改项目文件并在实例验证，最后报告改动及结果；无需访问 Windows 或执行部署脚本。
- 不擅自删除数据、模型、检查点、缓存、大目录或 Git 历史；不执行 `reset --hard`、`clean -fd`、force push，除非用户明确要求。

## Python 与运行验证

- 远程先激活 `/root/autodl-tmp/.venvs/haoyi_comb2_py313/bin/activate`，再使用该环境的 `python` 和 `python -m pip`。新任务先检查解释器及所需依赖，不混用环境。
- 本次 CLI 安装会话实测该远程解释器为 Python 3.13.15。项目 AGENTS.md 内的历史 combo2 版本不代表当前安装版本；更新前检查 `/root/comb_pkg/` 实际 wheel，并核实环境中的包版本。
- 2026-09-30 会话已核对实例 `combo2==1.1.3`；后续以实际环境为准。Windows 解释器路径不适用于实例；只有用户明确要求本地 Python 操作时才使用历史副本的 `.venv\Scripts\python.exe`。
- 不把历史运行结果当成当前实验状态；需要时读取实例内对应实验的实际日志、配置和指标。

## 项目入口与研究规范

- 项目连接因子/Memmap 数据、源级降维、模型训练预测、opt2 回测和评估。先读项目 `README.md`、`config.human`、`opt1_parameters.md`；发布版本看 `VERSION`、`RELEASE.md`。
- 常用入口：`runCombo.py`、`runEval.py`；其他评估模式在 `evals/comb_eval/run_eval_other.py`。示例位于 `eg-torch/`、`eg-lgbm/`，研究内容在 `haoyi_models/`。路径存在性与实际接口以文件为准。
- README 对后续新实验要求：先选成熟目标样本，再固定留出最新 21 个不同逻辑交易日作为验证；同日各时点不跨训练/验证；按完整目标依赖做 purge。验证不参与梯度更新。
- 每 epoch 以 eval/no_grad 计算 `val_ic_raw5w`：最终 Alpha 与未预处理的五日加权 Reference（权重 `[5,4,3,2,1]`，不除以 15）的有效股票 Pearson；先时点等权，再日期等权。严格改善，`min_delta=0`，连续 5 轮未改善停止，最多 25 轮；checkpoint 按实验声明选择。
- 以上为当前 README 的规范摘要。运行中、已冻结的旧实验保留其原规则，不因这份记忆自动改动。实施前核对 `haoyi_models/UNIFIED_RESEARCH_BENCHMARK_PROTOCOL_20260912.md` 及具体实验规范。

## 产物、同步和清理

- 实例运行数据、缓存、模型、预测、指标、日志、运行配置、审计、自动报告和压缩包长期只保留在实例对应项目/实验目录。
- 默认直接在实例分析。只有用户明确要求本地分析时才临时下载必要产物，并记录准确范围，任务结束前清理该任务的本地临时副本和派生文件；不要删除实例原件或既有用户文件。
- 图件、HTML 阅读页、绘图脚本及说明默认放在实例 `/root/autodl-tmp/haoyi_comb2_organize/artifacts/<任务>/`。只有用户明确要求本地撰写的交付物时才使用 `D:\Codex\haoyi_comb2_organize\artifacts\<任务>/`；不要为绘图新建或使用 docs 目录。
- 历史部署脚本 `D:\Codex\deploy_comb2_to_autodl.sh` 会用本地副本覆盖实例文件，且使用 rsync `--delete-delay`；不得作为日常流程自动运行。只有用户明确要求从本地恢复或导入代码时，才检查差异并按指定范围处理。其 `--exclude='/.tools/'` 仍须保留，以保护实例 Codex 的安装、配置及登录状态。

## 本次确定的 Codex CLI 使用约定

- CLI 根目录：`/root/autodl-tmp/haoyi_comb2_organize/.tools/codex-cli/`。
- `env.sh` 提供项目内的 Node/npm/CLI 路径；`CODEX_HOME` 为该目录下的 `home/`；临时目录为 `tmp/`。登录凭据属于敏感状态，不输出、不提交、不写入记忆。
- 本次安装并验证了官方 `@openai/codex` 0.159.0、Node.js v24.21.0；已使用 ChatGPT 登录。模型回复及终端 `pwd` 均已验证成功。这些版本/登录状态后续应按实际状态核实。
- 本实例是 Docker，禁止创建用户 namespace。0.159.0 的默认 bubblewrap 沙箱失败；单独测试的 0.158.0 Landlock 兼容路径也失败。不要反复下载旧版本或把“重装 bubblewrap”当成已经验证的修复。
- 用户最终明确选择：专用配置 `autodl-container` 使用 `sandbox_mode="danger-full-access"`、`approval_policy="on-request"`、`approvals_reviewer="auto_review"`。普通命令可直接执行，只有触发审批请求的操作才自动审核；没有“所有危险命令必定送审”的保证，也不是每条命令都审核。不要未经用户要求改成逐命令沙箱失败/提权重试的模式，或改为 `approval_policy="never"`。
- 专用配置文件为 `.tools/codex-cli/home/autodl-container.config.toml`；2026-09-30 排查桌面 SSH 对话仍使用沙箱后，也将上述三个权限字段写入 `.tools/codex-cli/home/config.toml`，并验证不带 profile 的 CLI 可直接执行命令。桌面端已有对话仍可能覆盖默认设置，需要在输入框下的权限菜单选择“自定义（config.toml）”；以实际会话下发的权限为准，不把配置文件等同于已生效的会话权限。裸 `codex` 沿用终端当前目录，不自动切换目录。
- 固定项目启动方式：`bash /root/autodl-tmp/haoyi_comb2_organize/.tools/codex-cli/start.sh`。该入口加载 env.sh、进入项目根并选择 autodl-container；用户仍可使用显式 `codex --cd /root/autodl-tmp/haoyi_comb2_organize --profile autodl-container`。
- 用中文简洁沟通，直接说明结果与真实限制；已授权范围内完成工作，不重复索要相同许可，不把 root 权限当成业务操作授权。没有用户明确要求时不创建额外子代理。
