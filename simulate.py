"""
Simulate hourly per-zone sensor streams.

Temperature = regional diurnal baseline + physically-motivated contributions:
  surface retention (concrete stores heat, releases it in the evening)
  low ventilation  (dense built-up blocks wind -> stagnant air)
  activity spike   (traffic / commerce heat, peaks 2-6 PM)
  low green cover  (no evapotranspiration / shade within ~300 m)
  cooling terms    (water, vegetation surplus)  + sensor noise

Because the data is simulated we know each contribution exactly, which gives
ground-truth cause labels to train and evaluate the classifier (TRL 3).
"""
import numpy as np
import pandas as pd
from config import CLIMATE

CAUSE_COLS = ["c_surface", "c_vent", "c_activity", "c_green"]
CAUSE_NAMES = ["Surface retention", "Low ventilation", "Activity spike", "Low green cover"]


def diurnal(h):
    dt = (h - 14) % 24                       # peak at 14:00, minimum at 05:00
    rise = np.sin(np.pi / 2 * (h - 5) / 9)
    k = np.exp(-15 / 6)
    fall = (np.exp(-dt / 6) - k) / (1 - k)
    return np.where((h >= 5) & (h <= 14), rise, fall)


def solar(h):
    return np.clip(np.sin(np.pi * (h - 6) / 12), 0, None)


def retention(h):
    d = (h - 20.5 + 12) % 24 - 12            # circular distance to 20:30
    return np.exp(-(d / 3.5) ** 2)


def activity_curve(h):
    return np.exp(-((h - 16) / 2.0) ** 2) + 0.4 * np.exp(-((h - 9) / 1.5) ** 2)


def fetch_live_weather(lat, lon):
    """Hourly regional baseline from Open-Meteo (free, no key). Returns DataFrame or None."""
    try:
        import requests
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
            "latitude": lat, "longitude": lon, "past_days": 7, "forecast_days": 1,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "wind_speed_unit": "ms", "timezone": "auto"})
        r.raise_for_status()
        j = r.json()["hourly"]
        w = pd.DataFrame({"time": pd.to_datetime(j["time"]), "t": j["temperature_2m"],
                          "rh": j["relative_humidity_2m"], "wind": j["wind_speed_10m"]}).dropna()
        w["day"] = (w.time.dt.normalize() - w.time.dt.normalize().min()).dt.days
        w["hour"] = w.time.dt.hour
        return w
    except Exception as e:  # offline, blocked, API change...
        print(f"  [live weather unavailable: {e.__class__.__name__}] -> using synthetic climate")
        return None


def simulate(zones, days=14, seed=11, live=None):
    rng = np.random.default_rng(seed)
    Z = len(zones)
    hours = np.arange(24)
    recs = []
    conc, green300 = zones.concrete.values, zones.green_300m.values
    water, soil = zones.water.values, zones.soil.values
    built, act = zones.built_up.values, zones.activity.values
    zone_bias = rng.normal(0, 0.15, Z)       # persistent micro-site effects
    # green-deficit reference adapts to the area (OSM maps far less green than exists in reality)
    g_ref = float(np.clip(np.quantile(green300, 0.75), 0.08, 0.30))

    if live is not None:
        days = int(live.day.max()) + 1

    for d in range(days):
        if live is None:
            tmin = rng.normal(CLIMATE["tmin"], 0.8)
            tmax = rng.normal(CLIMATE["tmax"], 1.3)
            wday = rng.uniform(CLIMATE["wind_lo"], CLIMATE["wind_hi"])
        for h in hours:
            if live is not None:
                row = live[(live.day == d) & (live.hour == h)]
                if row.empty:
                    continue
                base_t, base_rh, wb = row.t.iloc[0], row.rh.iloc[0], max(row.wind.iloc[0], 0.3)
            else:
                base_t = tmin + (tmax - tmin) * diurnal(h)
                base_rh = CLIMATE["rh_night"] - (CLIMATE["rh_night"] - CLIMATE["rh_day"]) * diurnal(h)
                wb = wday * (0.6 + 0.6 * solar(h))
            s, r, a = solar(h), retention(h), activity_curve(h)

            wind = np.clip(wb * (1 - 0.65 * built) * (1 + rng.normal(0, 0.08, Z)), 0.2, None)
            c_surface = 3.0 * conc * (0.45 * s + 1.0 * r)
            c_vent = 1.8 * np.clip(1.8 - wind, 0, None) * (0.5 + 0.5 * max(s, r))
            c_activity = 2.8 * act * (a + 0.15)
            c_green = 9.0 * (0.30 / g_ref) * np.clip(g_ref - green300, 0, None) * (0.4 + 0.6 * s)
            cooling = (-2.5 * water * (0.5 + 0.5 * s)
                       - 3.0 * np.clip(green300 - g_ref, 0, None) * (0.4 + 0.6 * s)
                       + soil * (0.9 * s - 0.6 * (1 - s)))
            temp = (base_t + c_surface + c_vent + c_activity + c_green + cooling
                    + zone_bias + rng.normal(0, 0.25, Z))
            rh = np.clip(base_rh + 12 * green300 + 18 * water - 6 * conc
                         - 1.5 * (temp - base_t) + rng.normal(0, 1.5, Z), 10, 100)
            recs.append(pd.DataFrame({
                "zone_id": zones.zone_id.values, "day": d, "hour": h,
                "temp": temp, "rh": rh, "wind": wind,
                "c_surface": c_surface, "c_vent": c_vent,
                "c_activity": c_activity, "c_green": c_green}))
    df = pd.concat(recs, ignore_index=True)
    return df.merge(zones[["zone_id", "concrete", "green", "water", "soil", "green_300m",
                           "built_up", "activity"]], on="zone_id")


def ground_truth_cause(df, mixed_ratio=0.9):
    """Dominant positive contribution; 'Mixed' when the runner-up is close."""
    C = df[CAUSE_COLS].values
    order = np.argsort(-C, axis=1)
    top = C[np.arange(len(C)), order[:, 0]]
    second = C[np.arange(len(C)), order[:, 1]]
    lab = np.array(CAUSE_NAMES)[order[:, 0]].astype(object)
    lab[second >= mixed_ratio * top] = "Mixed"
    return lab
