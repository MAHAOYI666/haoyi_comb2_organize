# 项目结构与兼容入口

`pyproject.toml` 定义发行包、开发依赖、命令、包映射和测试路径。源码安装与受保护 wheel 共用运行依赖和命令定义。VERSION 仍为唯一版本来源。

- `combo2/config.py`：XML 配置解析与校验。
- `combo2/paths.py`：实验 Name 校验与统一输出路径。
- `combo2/runtime.py`：Node、ExperimentRunner、训练预测回测流程和共享 IC 计算。
- `combo2/monitoring.py`：监控、计时、日志辅助组件。
- `combo2/cli/run_combo.py`：命令行解析与日志重定向。
- `combo2/bootstrap.py`：未安装源码运行时的一处兼容路径引导；安装包无需依赖仓库布局。
- `evals/comb_eval/`：评估服务，可以调用运行层；运行层不反向导入评估或 CLI。
- `vendor/comb2*`：保留现有公开包名和研究员接口，通过元数据映射安装。
- `config.py`、`runCombo.py`、`vendor/perf_monitor.py`：旧导入路径的兼容转发。
- `optuna_framework/`：仅源码研究工具，不包含在发布包或默认测试集合中。

保留历史配置示例用于迁移。无仓库内调用、未被打包的旧 CVXPY optimizer.py 已移除。实验输出、现有检查点和本地数据不属于本次清理范围。

Name 的行为和输出目录见根 README 与 config.human。跨实验的隔离由 Name 目录保证，目录内部保留回测和评估的标准文件名。
