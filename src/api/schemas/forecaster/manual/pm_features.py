import pandas as pd

def add_pm_features(df: pd.DataFrame, given_data: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["pm2_5_lag_12h"] = given_data["pm_2_5_lag_12h"].values[0]
    df["pm2_5_lag_24h"] = given_data["pm_2_5_lag_24h"].values[0]
    df["pm2_5_lag_48h"] = given_data["pm_2_5_lag_48h"].values[0]

    recent_pm = ["pm2_5_lag_12h","pm2_5_lag_24h","pm2_5_lag_48h",]

    df["pm2_5_change_12h"] = (df["pm2_5_lag_12h"] - df["pm2_5_lag_24h"])

    df["pm2_5_change_24h"] = (df["pm2_5_lag_24h"] - df["pm2_5_lag_48h"])

    df["pm2_5_acceleration"] = (df["pm2_5_lag_12h"] - 2 * df["pm2_5_lag_24h"]+ df["pm2_5_lag_48h"])

    df["pm2_5_recent_mean"] = df[recent_pm].mean(axis=1)
    df["pm2_5_recent_max"] = df[recent_pm].max(axis=1)

    past_pm25 = df["pm2_5"].shift(1)

    df["pm2_5_rolling_mean_24h"] = (past_pm25.rolling(window=4).mean())

    df["pm2_5_rolling_mean_48h"] = (past_pm25.rolling(window=8).mean())
    
    df.drop(columns=['pm2_5'], inplace=True)

    df['city'] = df['city'].astype('category')
    return df