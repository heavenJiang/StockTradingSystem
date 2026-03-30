# StockTradingSystem
StockTradingSystem,use AI


AI股票投资模型项目实施计划
该项目旨在开发一个基于AI模型的股票交易策略程序，核心包含数据获取、模型训练、历史回测以及Web UI交互功能。

Proposed Changes
依赖配置与项目基础
将使用 requirements.txt 管理依赖：

akshare: 获取A股历史K线数据。
pandas, numpy: 数据处理与指标运算。
scikit-learn: 构建AI模型（本次计划采用 RandomForestClassifier，因其对表格型数据有良好的分类预测性能，可以用于预测涨跌概率）。
streamlit: 构建现代化、美观的用户交互Web页面。
plotly: 用于UI中的图表绘制。
[NEW] 配置模块
[NEW] config.py
处理配置的加载和保存。配置项将保存在本地的 config.json 文件中。包含：

rebuild_model: bool (是否重新训练)
target_stock_code: str (如 "000001;600000")
backtest_year: int (回测年数)
stop_loss_threshold: float (如 0.1)
stop_profit_threshold: float (如 0.1)
model_path: str (模型保存目录)
init_capital: float (如 1000000.0)
max_hold_days: int (如 20)
[NEW] 模型构建模块
[NEW] buildmodel.py
数据获取: 使用 akshare.stock_zh_a_hist 获取日K线数据。
特征工程: 构建技术指标特征，如 MACD, 移动平均线 (MA5, MA10, MA20), RSI 等。
标签生成: 根据设定的持仓天数和止盈止损条件，判断未来N天内是否能达到止盈且未触碰止损，或者期满收益为正，作为正样本（1=买入），否则为负样本（0=不买入）。
模型训练: 依照 config.rebuild_model 的设置，选择重新训练 RandomForestClassifier 并通过 joblib 保存，或是加载本地模型。
[NEW] 回测模块
[NEW] backtest.py
回测循环: 按时间顺序遍历K线数据，调用模型预测买入信号。
交易逻辑:
买入：模型发出买入信号且当前无持仓时，全仓买入。
卖出：触发止损（严格执行）、触发止盈、或达到最大持仓天数时卖出。考虑跌停无法卖出的情况（实际以当日最低价与收盘价评估，简化处理可假设以收盘价或跌停价卖出）。
统计分析: 计算总交易次数、胜率、投资回报率、平均持仓自然日等，生成交易记录 DataFrame。
[NEW] 主模块
[NEW] astock_AImodel.py
提供供UI调用的核心API，封装整个流程，包括：
run_pipeline(): 阅读配置，调用 buildmodel 准备模型，调用 backtest 执行回测，并返回结果与未来建议。
[NEW] UI交互模块
[NEW] ui.py
基于 Streamlit 构建。
侧边栏/设置页: 提供表单修改并保存 config.json。
主页面:
“开始回测”按钮。
回测完成后，展示各项统计指标（用大的 Metric 组件显示）。
展示交易记录表格 (st.dataframe)。
使用 Plotly 绘制资金曲线或收益图表。
展示对下一个交易日的买卖预测建议。
Verification Plan
Automated Tests
无强制自动化单元测试，但会编写日志记录以便追踪流程。

Manual Verification
运行 streamlit run ui.py 启动服务。
在UI界面修改配置，检查 config.json 是否同步更新。
点击“开始回测”，观察是否能正确拉取数据、训练模型并输出回测结果页。
验证统计结果和未来的交易建议是否显示正常。