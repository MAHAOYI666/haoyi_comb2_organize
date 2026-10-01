# 项目上下文

当前项目位于 `/home/mahaoyi/projects/haoyi_comb2_organize`；相关模型在 `/home/mahaoyi/projects/combo26q4`。两者最初从 AutoDL 迁移，目前完全按 Notebook 的实际环境工作。

完整约束以项目根 AGENTS.md 为准。先激活 `/home/mahaoyi/.local/bin/haoyi-env.sh`，使用 `/home/mahaoyi/.venvs/haoyi_comb2_py313`；数据入口为 `/mnt/cache`、`/mnt/factors/QsimPool`、`/mnt/factors/ZsimPool`，均只读。输出留在项目或实验目录，图件放 artifacts/<任务>/。

运行入口为项目 bin 内的 runCombo、runEval、combo-hello-world。Codex 入口 `/home/mahaoyi/.local/bin/codex`，主配置与记忆 `/home/mahaoyi/.codex/`；从具体项目根启动。历史安装快照与聊天记录不作为当前路径依据。

研究规则见 README.md、config.human、opt1_parameters.md 和各实验协议。保留成熟目标、purge、末尾 21 个逻辑交易日验证、val_ic_raw5w、5 轮早停、最多 25 轮等新实验规则；冻结实验保持原口径。默认不启动训练，不改写历史模型、预测或源码快照。不添加助手署名或额外报告，临时测试文件用后清理。
