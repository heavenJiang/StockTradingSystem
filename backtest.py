import pandas as pd
import datetime
from config import cfg

class BacktestEnv:
    def __init__(self, stock_code, df_data, model, scaler):
        self.stock_code = stock_code
        self.df = df_data.reset_index(drop=True)
        self.model = model
        self.scaler = scaler
        
        self.init_capital = float(cfg.get("init_capital", 1000000.0))
        self.max_hold_days = int(cfg.get("max_hold_days", 20))
        self.stop_profit = float(cfg.get("stop_profit_threshold", 0.10))
        self.stop_loss = float(cfg.get("stop_loss_threshold", 0.10))
        
        self.capital = self.init_capital
        self.position = 0
        self.buy_price = 0.0
        self.buy_date = None
        self.hold_days = 0
        
        self.trades = []
        
    def step(self, current_day_idx):
        if current_day_idx >= len(self.df):
            return
            
        today = self.df.iloc[current_day_idx]
        
        # Checking hold status
        if self.position > 0:
            self.hold_days += 1
            
            # Check sell conditions (Simulate with day's price, using high/low)
            high_profit = (today['high'] - self.buy_price) / self.buy_price
            low_loss = (today['low'] - self.buy_price) / self.buy_price
            
            sell_price = 0.0
            sell_reason = ""
            
            # 1. Check stop loss (Pessimistic: assume sold at exact stop loss, or closing price if lower)
            if low_loss <= -self.stop_loss:
                # 检查是否跌停 (假设跌停为 -10% 或者 -20%, 则可能只能以收盘价甚至第二天才卖出)
                # 简化处理：以买入价乘以(1-止损率)为卖出价或者以当日最低价与收盘价较高者保守估计
                # 由于需求提到: "无法以止损价格卖出，比如跌停，按真实价格卖出"
                sell_price = min(today['close'], self.buy_price * (1 - self.stop_loss))
                sell_reason = "止损"
            
            # 2. Check stop profit 
            elif high_profit >= self.stop_profit:
                sell_price = max(today['close'], self.buy_price * (1 + self.stop_profit))
                sell_reason = "止盈"
                
            # 3. Check max hold days
            elif self.hold_days >= self.max_hold_days:
                sell_price = today['close']
                sell_reason = "到期"
                
            if sell_reason:
                self._execute_sell(today['date'], sell_price, sell_reason)
                
        else:
            # Predict buy action
            feature_cols = ['open', 'close', 'high', 'low', 'volume', 'pct_change',
                   'MA5', 'MA10', 'MA20', 'VMA5', 'VMA10', 
                   'MACD', 'Signal_Line', 'MACD_Hist', 
                   'return_1d', 'return_3d', 'return_5d', 'volatility_10d']
                   
            # Requires all features to not be NaN
            if today[feature_cols].isnull().any():
                return
                
            X = pd.DataFrame([today[feature_cols]])
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            
            if pred == 1:
                self._execute_buy(today['date'], today['close'])
                
    def _execute_buy(self, date, price):
        # 假设全仓买入
        shares_to_buy = (self.capital * 0.999) // price  # 扣除一点滑点/手续费预期 (简化)
        if shares_to_buy > 0:
            self.position = shares_to_buy
            self.buy_price = price
            self.buy_date = date
            self.hold_days = 0
            # 不在买入时扣减资金，而在卖出时结算以简化利息/手续费计算
            
    def _execute_sell(self, date, price, reason):
        profit = (price - self.buy_price) * self.position
        profit_pct = (price - self.buy_price) / self.buy_price
        
        self.capital += profit
        
        trade_record = {
            "买入日期": self.buy_date.strftime("%Y-%m-%d") if isinstance(self.buy_date, datetime.datetime) else self.buy_date,
            "股票代码": self.stock_code,
            "股票名称": "None", # 可后续补充
            "买入价格": round(self.buy_price, 2),
            "卖出日期": date.strftime("%Y-%m-%d") if isinstance(date, datetime.datetime) else date,
            "卖出价格": round(price, 2),
            "本次交易盈亏金额": round(profit, 2),
            "本次交易盈利百分比": f"{profit_pct*100:.2f}%",
            "卖出原因": reason,
            "持仓周期(天)": self.hold_days
        }
        self.trades.append(trade_record)
        
        # Reset position
        self.position = 0
        self.buy_price = 0.0
        self.buy_date = None
        self.hold_days = 0

    def get_results(self):
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(self.trades)
        if total_trades > 0:
            win_trades = len([t for t in self.trades if t['卖出原因'] == '止盈' or float(t['本次交易盈利百分比'].strip('%')) > 0])
            win_rate = win_trades / total_trades
            avg_hold_days = sum([t['持仓周期(天)'] for t in self.trades]) / total_trades
        else:
            win_rate = 0.0
            avg_hold_days = 0.0
            
        roi = (self.capital - self.init_capital) / self.init_capital
        
        stats = {
            "总交易次数": total_trades,
            "赢率": win_rate,
            "历史交易总投资回报率": roi,
            "平均每次交易用时(日)": avg_hold_days,
            "最终资金": self.capital
        }
        
        return trades_df, stats

def run_backtest(stock_code, df_data, model, scaler):
    env = BacktestEnv(stock_code, df_data, model, scaler)
    for i in range(len(df_data)):
        env.step(i)
    return env.get_results()
