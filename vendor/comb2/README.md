# comb2 研究员接入

完整研究员接口见 [config.human](../../config.human)，可运行的 loader/model 示例见 [eg-torch](../../eg-torch)。

Python 的 ResearchLoader.data_requirements 声明普通 Memmap 来源。process_source 在原始块上按采样时点处理并降维；model_input_sources、model_target、model_validity_source 选择模型用途。

模型输入为普通张量 [tsDays, stock, feature]，样本为 (idx, ds, ti, x, y, w)。历史窗口取连续交易日的同一时点快照。predict(x_window, di=..., ti=...) 返回股票截面。

数据集按需读取并复用有界缓存；none/fp4/fp8 直接作用于张量。模型保存、加载和 trainii 股票索引回填保持现有接口。

runCombo config.xml 统一运行单时点和多时点实验；多时点回测使用 opt2。runEval config.xml 用研究员目标计算 IC，用实际日终资产计算 PnL。
