import streamlit as st
import pandas as pd
import json
from config import cfg
import astock_AImodel
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI 股票量化投资模型", page_icon="📈", layout="wide")

st.title("📈 AI 股票量化投资策略系统")
st.markdown("通过AI模型学习历史K线数据中的模式，实现自动化预测与回测验证，并对未来投资提供决策建议。")

# SIDEBAR CONFIGURATION
st.sidebar.header("🎯 系统参数配置")

with st.sidebar.form("config_form"):
    st.subheader("基础模型设置")
    rebuild_model = st.checkbox("重新训练 AI 模型", value=cfg.get("rebuild_model"))
    target_stock_code = st.text_input("目标股票代码 (多个用分号隔开)", value=cfg.get("target_stock_code"), help="例如: 000001;600000")
    backtest_year = st.number_input("回测年数", min_value=1, max_value=20, value=cfg.get("backtest_year"))
    
    st.subheader("交易策略阈值")
    stop_loss_threshold = st.number_input("止损阈值 (小数)", min_value=0.01, max_value=0.99, value=float(cfg.get("stop_loss_threshold")), step=0.01)
    stop_profit_threshold = st.number_input("止盈阈值 (小数)", min_value=0.01, max_value=2.0, value=float(cfg.get("stop_profit_threshold")), step=0.01)
    max_hold_days = st.number_input("最大持仓天数", min_value=1, max_value=200, value=int(cfg.get("max_hold_days")))
    
    st.subheader("资金与路径")
    init_capital = st.number_input("初始总资金 (元)", min_value=1000.0, value=float(cfg.get("init_capital")), step=1000.0)
    model_path = st.text_input("模型保存路径", value=cfg.get("model_path"))

    submit_button = st.form_submit_button("保存配置")

    if submit_button:
        new_config = {
            "rebuild_model": rebuild_model,
            "target_stock_code": target_stock_code,
            "backtest_year": int(backtest_year),
            "stop_loss_threshold": float(stop_loss_threshold),
            "stop_profit_threshold": float(stop_profit_threshold),
            "max_hold_days": int(max_hold_days),
            "init_capital": float(init_capital),
            "model_path": model_path
        }
        cfg.set_all(new_config)
        st.sidebar.success("配置已成功更新！")

# MAIN CONTENT
st.header("📊 回测与预测中心")

if st.button("🚀 开始模型训练与历史回测", use_container_width=True, type="primary"):
    with st.spinner("📦 正在拉取数据、训练模型并执行回测，请稍候..."):
        try:
            results = astock_AImodel.run_pipeline()
            st.session_state['results'] = results
            st.success("回测完成！")
        except Exception as e:
            st.error(f"执行出错: {e}")

if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Create tabs for each stock
    stock_codes = list(results.keys())
    tabs = st.tabs(stock_codes)
    
    for idx, (code, data) in enumerate(results.items()):
        with tabs[idx]:
            if data["status"] != "success":
                st.warning(f"处理失败: {data.get('error', '未知错误')}")
                continue
                
            st.subheader(f"💡 {code} 明日交易建议")
            st.info(data["suggestion"])
            
            stats = data["stats"]
            
            st.subheader("📝 回测统计分析")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总交易次数", f"{stats['总交易次数']} 次")
            col2.metric("模型胜率 (赢率)", f"{stats['赢率'] * 100:.2f}%")
            
            roi = stats['历史交易总投资回报率']
            roi_color = "normal" if roi > 0 else "inverse"
            col3.metric("总投资回报率 (ROI)", f"{roi * 100:.2f}%", delta=f"{roi*100:.2f}%", delta_color=roi_color)
            
            col4.metric("平均交易用时 (日)", f"{stats['平均每次交易用时(日)']:.1f} 天")
            
            st.metric("最终账户总资产", f"¥ {stats['最终资金']:.2f}")
            
            st.subheader("📒 详细交易记录")
            trades_df = data["trades_df"]
            if trades_df is not None and not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
                
                # Plot PnL distribution
                st.subheader("盈亏收益分布图")
                trades_df['本次交易盈亏金额'] = trades_df['本次交易盈亏金额'].astype(float)
                fig = px.histogram(trades_df, x="本次交易盈亏金额", nbins=20, 
                                   title=f"单次交易盈亏分布 - {code}", 
                                   color="卖出原因")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("未能产生任何有效交易（模型可能未发出买入信号）。")
