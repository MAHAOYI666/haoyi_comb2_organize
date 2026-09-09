# comb2 研究员接入

完整研究员接口见 [config.human](../../config.human)，可运行的 loader/model 示例见 [eg-torch](../../eg-torch)。

Python 的 ResearchLoader.data_requirements 声明普通 Memmap 来源。process_source 在原始块上按采样时点处理并降维；model_input_sources、model_target 选择模型用途。配置 cache_path 时默认使用 delay=1 的 BaseUnivMask、NoNewStockMask、LimitMask 交集，model_validity_source 可显式替换默认筛选。

模型输入为普通张量 [tsDays, stock, feature]，样本为 (idx, ds, ti, x, y, w)。历史窗口取连续交易日的同一时点快照。predict(x_window, di=..., ti=...) 返回股票截面。

训练数据集分块预加载各时点的特征和标签，取样时复用内存快照并执行窗口变换；none/fp4/fp8 直接作用于特征张量。训练存储随日期、时点、股票和特征数量增长，源数据更新后需重建数据集。模型保存、加载和 trainii 股票索引回填保持现有接口。

runCombo config.xml 统一运行单时点和多时点实验；多时点回测使用 opt2。runEval config.xml 用研究员目标计算 IC，用实际日终资产计算 PnL。
