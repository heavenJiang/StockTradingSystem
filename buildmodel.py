import os
import datetime
import pandas as pd
import numpy as np
import akshare as ak
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from config import cfg

def fetch_data(stock_code, start_date, end_date):
    """Fetch daily historical data from akshare"""
    try:
        # akshare expects stock code with prefix if needed, but stock_zh_a_hist usually expects 6 digits
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Rename columns standard
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_change"
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return pd.DataFrame()

def calculate_features(df):
    """Calculate technical indicators as features"""
    df = df.copy()
    
    # MA
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # Volume MA
    df['VMA5'] = df['volume'].rolling(window=5).mean()
    df['VMA10'] = df['volume'].rolling(window=10).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Return features
    df['return_1d'] = df['close'].pct_change(1)
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    
    # Volatility
    df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
    
    return df

def generate_labels(df, max_hold_days, stop_profit, stop_loss):
    """
    Generate target labels for training.
    1: Profitable within max_hold_days (hits stop_profit before stop_loss, or ends with >0 return)
    0: Not profitable
    """
    labels = []
    n = len(df)
    
    for i in range(n):
        if i + 1 >= n:
            labels.append(np.nan)
            continue
            
        buy_price = df['close'].iloc[i]  # Assume buying at close price of current day
        label = 0
        
        # Look forward max_hold_days
        for j in range(1, max_hold_days + 1):
            if i + j >= n:
                break
            
            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]
            future_close = df['close'].iloc[i + j]
            
            highest_profit = (future_high - buy_price) / buy_price
            lowest_loss = (future_low - buy_price) / buy_price
            
            # Check stop loss first (pessimistic)
            if lowest_loss <= -stop_loss:
                label = 0
                break
                
            # Check stop profit
            if highest_profit >= stop_profit:
                label = 1
                break
                
            # Reached max days
            if j == max_hold_days:
                if (future_close - buy_price) / buy_price > 0:
                    label = 1
                else:
                    label = 0
                    
        labels.append(label)
        
    df['label'] = labels
    return df

def build_model(stock_code):
    """Build AI model for a single stock"""
    print(f"Start building model for {stock_code}...")
    
    rebuild = cfg.get("rebuild_model", True)
    model_dir = cfg.get("model_path", "./models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_file = os.path.join(model_dir, f"{stock_code}_model.pkl")
    scaler_file = os.path.join(model_dir, f"{stock_code}_scaler.pkl")
    
    if not rebuild and os.path.exists(model_file) and os.path.exists(scaler_file):
        print(f"Loading existing model from {model_file}...")
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler

    # Fetch data: up to current date, start date from backtest_year + some buffer for features
    backtest_year = int(cfg.get("backtest_year", 3))
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=(backtest_year + 1) * 365) # +1 calc buffer
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    df = fetch_data(stock_code, start_str, end_str)
    if df.empty:
        print(f"Cannot fetch data for {stock_code}")
        return None, None
        
    # Feature Engineering
    df = calculate_features(df)
    
    # Generate Labels
    max_hold = int(cfg.get("max_hold_days", 20))
    stop_p = float(cfg.get("stop_profit_threshold", 0.10))
    stop_l = float(cfg.get("stop_loss_threshold", 0.10))
    
    df = generate_labels(df, max_hold, stop_p, stop_l)
    
    # Drop NaNs
    df = df.dropna()
    
    if len(df) < 50:
        print(f"Not enough data to train model for {stock_code}. len={len(df)}")
        return None, None
        
    # Features definition
    feature_cols = ['open', 'close', 'high', 'low', 'volume', 'pct_change',
                   'MA5', 'MA10', 'MA20', 'VMA5', 'VMA10', 
                   'MACD', 'Signal_Line', 'MACD_Hist', 
                   'return_1d', 'return_3d', 'return_5d', 'volatility_10d']
                   
    X = df[feature_cols]
    y = df['label']
    
    print(f"Training data size: {len(X)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model Training
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)
    
    # Save Model
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    print(f"Model saved to {model_file}")
    
    return model, scaler

if __name__ == "__main__":
    # Test script
    build_model("000001")
