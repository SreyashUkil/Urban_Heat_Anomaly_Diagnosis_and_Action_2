"""DETECT: anomaly = zone temperature vs city-wide reference at the same hour."""
import numpy as np
import pandas as pd
from config import WATCH_DT, HOT_DT


def add_anomaly(df):
    ref = df.groupby(["day", "hour"]).temp.transform("median")
    df["ref_temp"] = ref
    df["dT"] = df.temp - ref
    df["severity"] = np.select([df.dT >= HOT_DT, df.dT >= WATCH_DT], ["Hot", "Watch"], "Normal")
    return df


def zone_profiles(df):
    """Typical-day profile per zone (mean over all days) and zone-level severity."""
    typ = df.groupby(["zone_id", "hour"], as_index=False)[["temp", "dT", "rh", "wind"]].mean()
    peak = typ.loc[typ.groupby("zone_id").dT.idxmax()].rename(
        columns={"hour": "peak_hour", "dT": "peak_dT", "temp": "peak_temp",
                 "rh": "peak_rh", "wind": "peak_wind"})
    agg = typ.groupby("zone_id").agg(mean_temp=("temp", "mean"), mean_dT=("dT", "mean"),
                                     day_dT=("dT", lambda s: s.iloc[12:18].mean()),
                                     night_dT=("dT", lambda s: pd.concat([s.iloc[20:], s.iloc[:3]]).mean()))
    hot_hours = typ[typ.dT >= WATCH_DT].groupby("zone_id").hour.apply(_hour_window)
    z = peak.set_index("zone_id").join(agg).join(hot_hours.rename("hot_hours"))
    z["hot_hours"] = z.hot_hours.fillna("—")
    z["severity"] = np.select([z.peak_dT >= HOT_DT, z.peak_dT >= WATCH_DT], ["Hot", "Watch"], "Normal")
    return typ, z.reset_index()


def _hour_window(hours):
    hs = sorted(set(hours))
    if not hs:
        return "—"
    if len(hs) == 24:
        return "all day (24 h)"
    # group consecutive hours (circular)
    runs, start, prev = [], hs[0], hs[0]
    for h in hs[1:]:
        if h != prev + 1:
            runs.append((start, prev)); start = h
        prev = h
    runs.append((start, prev))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == 23:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return ", ".join(f"{a:02d}:00–{(b + 1) % 24:02d}:00" for a, b in runs)
