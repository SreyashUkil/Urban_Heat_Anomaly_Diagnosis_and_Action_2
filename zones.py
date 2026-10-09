"""Build the zone grid and assign (synthetic) land-use / urban-context features."""
import numpy as np
import pandas as pd
from config import AREA, PROFILES, LANDMARKS, RIVER

KM_PER_DEG = 111.0


def _km(lat, lon, lat0, lon0):
    dy = (lat - lat0) * KM_PER_DEG
    dx = (lon - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    return np.hypot(dx, dy)


def _dist_to_polyline(lat, lon, pts):
    lat0 = np.mean([p[0] for p in pts])
    c = np.cos(np.radians(lat0)) * KM_PER_DEG
    px, py = lon * c, lat * KM_PER_DEG
    best = np.full_like(lat, np.inf, dtype=float)
    for (a_lat, a_lon), (b_lat, b_lon) in zip(pts[:-1], pts[1:]):
        ax, ay, bx, by = a_lon * c, a_lat * KM_PER_DEG, b_lon * c, b_lat * KM_PER_DEG
        vx, vy = bx - ax, by - ay
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        best = np.minimum(best, np.hypot(px - (ax + t * vx), py - (ay + t * vy)))
    return best


def _smooth_noise(rng, rows, cols, passes=3):
    z = rng.normal(0, 1, (rows, cols))
    for _ in range(passes):  # cheap box blur, no scipy needed
        p = np.pad(z, 1, mode="edge")
        z = sum(p[i:i + rows, j:j + cols] for i in range(3) for j in range(3)) / 9
    return z / (z.std() + 1e-9)


def _neigh_mean(grid):
    r, c = grid.shape
    p = np.pad(grid, 1, mode="edge")
    return sum(p[i:i + r, j:j + c] for i in range(3) for j in range(3)) / 9


def build_zones(area=AREA, landmarks=LANDMARKS, river=RIVER, seed=7):
    rng = np.random.default_rng(seed)
    R, C = area["rows"], area["cols"]
    dlat = (area["lat_max"] - area["lat_min"]) / R
    dlon = (area["lon_max"] - area["lon_min"]) / C
    lat = area["lat_max"] - dlat * (np.arange(R) + 0.5)        # row 0 = north
    lon = area["lon_min"] + dlon * (np.arange(C) + 0.5)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    keys = ["concrete", "green", "water", "soil", "built_up", "activity"]
    F = {k: np.full((R, C), v) for k, v in zip(keys, PROFILES["base"])}

    seeds = list(landmarks)
    if river:
        d = _dist_to_polyline(LAT, LON, river["points"])
        w = np.exp(-(d / river["width_km"]) ** 2) * 0.95
        for k, v in zip(keys, PROFILES[river["profile"]]):
            F[k] = F[k] * (1 - w) + v * w
    for _, la, lo, prof, rad, strength in seeds:
        w = strength * np.exp(-(_km(LAT, LON, la, lo) / rad) ** 2)
        for k, v in zip(keys, PROFILES[prof]):
            F[k] = F[k] * (1 - w) + v * w

    # spatially-correlated variation so zones are not perfectly smooth
    F["concrete"] += 0.07 * _smooth_noise(rng, R, C)
    F["green"] += 0.05 * _smooth_noise(rng, R, C)
    F["activity"] += 0.06 * _smooth_noise(rng, R, C)
    for k in keys:
        F[k] = np.clip(F[k], 0.0, 1.0)
    tot = F["concrete"] + F["green"] + F["water"] + F["soil"]
    for k in ["concrete", "green", "water", "soil"]:
        F[k] = F[k] / tot

    green300 = _neigh_mean(F["green"])  # ~ green cover within 300 m

    rows, cols = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    df = pd.DataFrame({
        "zone_id": [f"Z{r:02d}{c:02d}" for r, c in zip(rows.ravel(), cols.ravel())],
        "row": rows.ravel(), "col": cols.ravel(),
        "lat": LAT.ravel(), "lon": LON.ravel(),
        "concrete": F["concrete"].ravel(), "green": F["green"].ravel(),
        "water": F["water"].ravel(), "soil": F["soil"].ravel(),
        "green_300m": green300.ravel(),
        "built_up": F["built_up"].ravel(), "activity": F["activity"].ravel(),
    })
    df.attrs.update(dlat=dlat, dlon=dlon, rows=R, cols=C)

    # nearest landmark name, for human-readable reports
    names = [l[0] for l in seeds] + ([river["label"]] if river else [])
    dists = [_km(df.lat.values, df.lon.values, l[1], l[2]) for l in seeds]
    if river:
        dists.append(_dist_to_polyline(df.lat.values, df.lon.values, river["points"]))
    D = np.vstack(dists)
    df["near"] = [names[i] for i in D.argmin(0)]
    df["near_km"] = D.min(0)
    return df


def random_landmarks(area, n=9, seed=7):
    """For a custom area (no known landmarks): random land-use seeds."""
    rng = np.random.default_rng(seed)
    profs = ["dense_core", "market", "transit_hub", "residential", "residential",
             "campus_green", "water_park", "rocky_scrub", "rail_yard"]
    out = []
    for i in range(n):
        la = rng.uniform(area["lat_min"], area["lat_max"])
        lo = rng.uniform(area["lon_min"], area["lon_max"])
        p = profs[i % len(profs)]
        out.append((f"Seed {i+1}: {p.replace('_', ' ')}", la, lo, p, rng.uniform(0.35, 0.8), 0.85))
    return out
