"""
Study-area configuration.

Default area: Vellore–Katpadi (Tamil Nadu). Landmark positions are APPROXIMATE
and the land-use they seed is SYNTHETIC (TRL 3). Replace with OSM / Sentinel-2 /
Landsat-derived layers when moving to real-data validation.
"""

AREA = {
    "name": "Vellore – Katpadi",
    "lat_min": 12.905, "lat_max": 12.985,
    "lon_min": 79.105, "lon_max": 79.185,
    "rows": 28, "cols": 28,          # 28 x 28 = 784 zones (~300 m each)
}

# Land-use profiles: fractions of the zone surface + context indices (0-1)
PROFILES = {
    #               concrete green water soil  built activity
    "base":        (0.38, 0.27, 0.00, 0.35, 0.38, 0.25),
    "dense_core":  (0.82, 0.05, 0.00, 0.13, 0.85, 0.80),
    "market":      (0.85, 0.03, 0.00, 0.12, 0.80, 1.00),
    "transit_hub": (0.80, 0.05, 0.00, 0.15, 0.55, 0.95),
    "rail_yard":   (0.84, 0.14, 0.00, 0.02, 0.40, 0.30),
    "residential": (0.55, 0.12, 0.00, 0.33, 0.88, 0.30),
    "campus_green":(0.28, 0.55, 0.02, 0.15, 0.30, 0.30),
    "water_park":  (0.15, 0.35, 0.40, 0.10, 0.15, 0.20),
    "river_sand":  (0.03, 0.10, 0.07, 0.80, 0.03, 0.05),
    "rocky_scrub": (0.05, 0.25, 0.00, 0.70, 0.05, 0.05),
}

# (label, lat, lon, profile, radius_km, strength)  -- approximate positions
LANDMARKS = [
    ("Vellore Fort & moat",          12.9209, 79.1302, "water_park",  0.35, 0.95),
    ("CMC / old town core",          12.9248, 79.1360, "dense_core",  0.65, 0.90),
    ("Long Bazaar market",           12.9228, 79.1328, "market",      0.30, 0.90),
    ("New Bus Stand",                12.9340, 79.1378, "transit_hub", 0.35, 0.90),
    ("Katpadi Junction rail yard",   12.9716, 79.1378, "rail_yard",   0.45, 0.90),
    ("Katpadi main road strip",      12.9625, 79.1405, "dense_core",  0.40, 0.75),
    ("Gandhi Nagar (residential)",   12.9545, 79.1360, "residential", 0.55, 0.70),
    ("Sathuvachari (residential)",   12.9360, 79.1650, "residential", 0.65, 0.70),
    ("VIT campus",                   12.9692, 79.1559, "campus_green",0.60, 0.90),
    ("Western rocky scrub (approx.)",12.9150, 79.1130, "rocky_scrub", 0.90, 0.80),
]

# Palar riverbed (mostly dry sand) as a polyline -- approximate
RIVER = {
    "label": "Palar riverbed",
    "profile": "river_sand",
    "width_km": 0.30,
    "points": [(12.9490, 79.1050), (12.9465, 79.1250), (12.9440, 79.1450),
               (12.9455, 79.1650), (12.9490, 79.1850)],
}

# Synthetic climate used when --live weather is not available (pre-summer, Vellore-like)
CLIMATE = {"tmin": 27.0, "tmax": 37.5, "rh_night": 70.0, "rh_day": 38.0,
           "wind_lo": 1.5, "wind_hi": 4.5}

# Anomaly thresholds on dT = zone temp - city median at the same hour (deg C)
WATCH_DT = 0.8
HOT_DT = 1.8

CAUSES = ["Surface retention", "Low ventilation", "Activity spike", "Low green cover", "Mixed"]
CAUSE_COLORS = {
    "Surface retention": "#d7301f",
    "Low ventilation":   "#7b3294",
    "Activity spike":    "#f39c12",
    "Low green cover":   "#8c6d31",
    "Mixed":             "#4a4a4a",
    "None":              "#eeeeee",
}
