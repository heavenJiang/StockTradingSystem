import os
from config import cfg
import buildmodel
import backtest
import datetime

def run_pipeline():
    """Run model build and backtest for all target stocks"""
    target_stocks = cfg.get("target_stock_code", "").split(";")
    target_stocks = [s.strip() for s in target_stocks if s.strip()]
    
    results = {}
    
    for stock_code in target_stocks:
        print(f"========= Processing {stock_code} =========")
        # 1. Build Model
        model, scaler = buildmodel.build_model(stock_code)
        
        if model is None or scaler is None:
            results[stock_code] = {"status": "failed", "error": "Model build failed or not enough data"}
            continue
            
        # 2. Backtest
        print(f"Starting backtest for {stock_code}...")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=(int(cfg.get("backtest_year", 3)) * 365))
        
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        
        df = buildmodel.fetch_data(stock_code, start_str, end_str)
        df = buildmodel.calculate_features(df)
        df = df.dropna().reset_index(drop=True)
        
        trades_df, stats = backtest.run_backtest(stock_code, df, model, scaler)
        
        # 3. Predict next day action
        last_day = df.iloc[-1]
        feature_cols = ['open', 'close', 'high', 'low', 'volume', 'pct_change',
               'MA5', 'MA10', 'MA20', 'VMA5', 'VMA10', 
               'MACD', 'Signal_Line', 'MACD_Hist', 
               'return_1d', 'return_3d', 'return_5d', 'volatility_10d']
               
        X_last = [last_day[feature_cols]]
        X_last_scaled = scaler.transform(X_last)
        pred = model.predict(X_last_scaled)[0]
        
        suggestion = ""
        stop_p = cfg.get("stop_profit_threshold")
        if pred == 1:
            suggestion = f"建议在下一个交易日以{last_day['close']:.2f}附近价格买入股票 {stock_code}，预期收益 {stop_p*100:.2f}%。"
        else:
            suggestion = f"建议在下一个交易日继续观望，不买入 {stock_code}。"
            
        results[stock_code] = {
            "status": "success",
            "trades_df": trades_df,
            "stats": stats,
            "suggestion": suggestion
        }
        print(f"Completed {stock_code}.")
        
    return results

if __name__ == "__main__":
    res = run_pipeline()
    for code, data in res.items():
        if data["status"] == "success":
            print(f"[{code}] Total trades: {data['stats']['总交易次数']}, Win Rate: {data['stats']['赢率']:.2f}, ROI: {data['stats']['历史交易总投资回报率']:.2f}")
            print(f"Suggestion: {data['suggestion']}")
