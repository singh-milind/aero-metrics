import pandas as pd

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df['pm_ratio'] = df['pm10'] / df['pm2_5']
    df["pm_ratio_lag_12h"] = df["pm_ratio"].shift(2)
    df["pm_ratio_lag_24h"] = df["pm_ratio"].shift(4)
    df["pm_ratio_lag_48h"] = df["pm_ratio"].shift(8)

    recent_pm = ["pm_ratio_lag_12h","pm_ratio_lag_24h","pm_ratio_lag_48h",]

    df["pm_ratio_change_12h"] = (df["pm_ratio_lag_12h"] - df["pm_ratio_lag_24h"])

    df["pm_ratio_change_24h"] = (df["pm_ratio_lag_24h"] - df["pm_ratio_lag_48h"])

    df["pm_ratio_acceleration"] = (df["pm_ratio_lag_12h"] - 2 * df["pm_ratio_lag_24h"]+ df["pm_ratio_lag_48h"])

    df["pm_ratio_recent_mean"] = df[recent_pm].mean(axis=1)
    df["pm_ratio_recent_max"] = df[recent_pm].max(axis=1)

    past_pm25 = df["pm_ratio"].shift(1)

    df["pm_ratio_rolling_mean_24h"] = (past_pm25.rolling(window=4).mean())

    df["pm_ratio_rolling_mean_48h"] = (past_pm25.rolling(window=8).mean())
    
    df.drop(columns=['pm_ratio'], inplace=True)

    df['city'] = df['city'].astype('category')
    return df