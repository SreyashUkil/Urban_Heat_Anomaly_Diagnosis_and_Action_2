"""
Real land use from OpenStreetMap via the Overpass API (free, no key).

Fetches buildings, roads, parks/trees, water, sand, commercial/industrial areas and
named places inside the study bounding box, rasterises them onto a fine grid
(~10 m pixels) and aggregates to each zone:

    concrete   = building footprint + paved surface (roads, parking, rail)
    green      = trees / woods / parks (+ 0.6 x grass, meadow, farmland)
    water      = water bodies and river polygons
    soil       = everything else (open ground, sand, unmapped)
    built_up   = building coverage (0.45 coverage => 1.0)
    activity   = traffic-weighted road length + commercial / market / transit / industrial area

Responses are cached in ./osm_cache so later runs (and your demo) work offline.
"""
import hashlib
import json
import os
import time
import numpy as np
from PIL import Image, ImageDraw

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osm_cache")

QUERY = """
[out:json][timeout:180];
(
  way["building"]({b});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|motorway_link|trunk_link|primary_link|secondary_link)$"]({b});
  way["amenity"~"^(parking|marketplace|bus_station)$"]({b});
  way["railway"~"^(rail|platform)$"]({b});
  way["landuse"~"^(commercial|retail|industrial|railway|grass|forest|meadow|orchard|recreation_ground|cemetery|village_green|reservoir|basin|farmland|plant_nursery)$"]({b});
  relation["landuse"~"^(commercial|retail|industrial|forest|reservoir|farmland|grass)$"]({b});
  way["leisure"~"^(park|garden|pitch|golf_course|nature_reserve|playground)$"]({b});
  relation["leisure"~"^(park|garden|nature_reserve)$"]({b});
  way["natural"~"^(wood|scrub|grassland|water|wetland|sand|bare_rock|tree_row|heath)$"]({b});
  relation["natural"~"^(wood|water|scrub|sand|wetland)$"]({b});
  way["waterway"~"^(river|riverbank|canal|stream)$"]({b});
  relation["waterway"~"^(river|riverbank)$"]({b});
  way["water"]({b});
  node["natural"="tree"]({b});
  node["amenity"~"^(marketplace|bus_station|hospital|university|college)$"]["name"]({b});
  node["railway"="station"]["name"]({b});
  node["place"~"^(suburb|neighbourhood|quarter|village|hamlet|town)$"]["name"]({b});
);
out geom;
"""

ROAD_WIDTH_M = {"motorway": 20, "trunk": 16, "primary": 13, "secondary": 10, "tertiary": 8,
                "unclassified": 6, "residential": 5, "living_street": 4, "service": 4}
ROAD_TRAFFIC = {"motorway": 1.0, "trunk": 1.0, "primary": 0.9, "secondary": 0.65, "tertiary": 0.45,
                "unclassified": 0.25, "residential": 0.15, "living_street": 0.1, "service": 0.08}
GREEN_FULL = {("natural", "wood"), ("landuse", "forest"), ("leisure", "park"), ("leisure", "garden"),
              ("leisure", "nature_reserve"), ("natural", "scrub"), ("landuse", "orchard"),
              ("landuse", "cemetery"), ("landuse", "plant_nursery"), ("natural", "tree_row")}
GREEN_PART = {("landuse", "grass"), ("landuse", "meadow"), ("natural", "grassland"), ("landuse", "farmland"),
              ("landuse", "village_green"), ("landuse", "recreation_ground"), ("leisure", "pitch"),
              ("leisure", "golf_course"), ("leisure", "playground"), ("natural", "heath"), ("natural", "wetland")}


# ----------------------------------------------------------------------------- fetch
def _bbox_key(area):
    s = f"{area['lat_min']:.4f},{area['lon_min']:.4f},{area['lat_max']:.4f},{area['lon_max']:.4f}"
    return s, hashlib.md5(s.encode()).hexdigest()[:10]


def fetch_osm(area, refresh=False):
    """Return Overpass JSON for the area (from cache if available)."""
    bbox, key = _bbox_key(area)
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"overpass_{key}.json")
    if os.path.exists(path) and not refresh:
        print(f"  using cached OSM data ({os.path.basename(path)})")
        return json.load(open(path, encoding="utf-8"))
    import requests
    q = QUERY.format(b=bbox)
    last = None
    for url in ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"  querying {url.split('/')[2]} … (can take 30–90 s)")
                r = requests.post(url, data={"data": q}, timeout=200,
                                  headers={"User-Agent": "urban-heat-anomaly-student-project/1.0"})
                if r.status_code in (429, 504):
                    last = f"HTTP {r.status_code} (server busy)"
                    time.sleep(10); continue
                r.raise_for_status()
                data = r.json()
                json.dump(data, open(path, "w", encoding="utf-8"))
                print(f"  received {len(data.get('elements', [])):,} OSM features → cached")
                return data
            except Exception as e:
                last = f"{e.__class__.__name__}: {e}"[:160]
                break
    raise RuntimeError(f"Overpass unavailable ({last})")


def load_osm_file(path):
    return json.load(open(path, encoding="utf-8"))


# ----------------------------------------------------------------------------- rasterise
def _shapes(el):
    """Yield (coords, closed) lists of (lat, lon) for a way or a relation's outer members."""
    if el["type"] == "way" and "geometry" in el:
        g = [(p["lat"], p["lon"]) for p in el["geometry"]]
        yield g, len(g) > 2 and g[0] == g[-1]
    elif el["type"] == "relation":
        for m in el.get("members", []):
            if m.get("role", "outer") in ("outer", "") and "geometry" in m:
                g = [(p["lat"], p["lon"]) for p in m["geometry"]]
                yield g, True   # treat member rings as areas


def _classify(tags):
    """Map OSM tags to (layer, extra) for rasterisation."""
    if "building" in tags:
        return "building", None
    hw = tags.get("highway")
    if hw:
        return "road", hw.replace("_link", "")
    if tags.get("railway") in ("rail", "platform") or tags.get("landuse") == "railway":
        return "rail", None
    if tags.get("amenity") == "parking":
        return "paved", None
    if tags.get("amenity") in ("marketplace", "bus_station"):
        return "hub", None
    if (tags.get("natural") == "water" or "water" in tags or tags.get("landuse") in ("reservoir", "basin")
            or tags.get("waterway") in ("riverbank",)):
        return "water", None
    if tags.get("waterway") in ("river", "canal", "stream"):
        return "waterline", tags.get("waterway")
    if tags.get("natural") in ("sand", "bare_rock"):
        return "sand", None
    if tags.get("landuse") in ("commercial", "retail"):
        return "commercial", None
    if tags.get("landuse") == "industrial":
        return "industrial", None
    for k in ("natural", "landuse", "leisure"):
        if (k, tags.get(k)) in GREEN_FULL:
            return "green", None
        if (k, tags.get(k)) in GREEN_PART:
            return "grass", None
    return None, None


def osm_to_zone_features(data, area, sub=30):
    """Rasterise OSM features and aggregate to the R x C zone grid."""
    R, C = area["rows"], area["cols"]
    H, W = R * sub, C * sub
    lat0, lat1, lon0, lon1 = area["lat_min"], area["lat_max"], area["lon_min"], area["lon_max"]
    m_per_px = (lat1 - lat0) * 111_000 / H

    def px(lat, lon):
        return ((lon - lon0) / (lon1 - lon0) * W, (lat1 - lat) / (lat1 - lat0) * H)

    layers = {k: Image.new("L", (W, H), 0) for k in
              ["building", "paved", "green", "grass", "water", "sand", "commercial", "industrial", "hub"]}
    traffic = np.zeros((H, W), dtype=np.float32)
    draw = {k: ImageDraw.Draw(v) for k, v in layers.items()}
    names, n_by_layer = [], {}

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            if tags.get("natural") == "tree":           # single tree ≈ 6 m crown
                x, y = px(el["lat"], el["lon"]); r = max(0.5, 3 / m_per_px)
                draw["green"].ellipse([x - r, y - r, x + r, y + r], fill=255)
                n_by_layer["tree"] = n_by_layer.get("tree", 0) + 1
            elif "name" in tags:
                kind = tags.get("place") or tags.get("amenity") or ("station" if tags.get("railway") else "poi")
                names.append((tags["name"], el["lat"], el["lon"], kind))
            continue
        layer, extra = _classify(tags)
        if layer is None:
            continue
        n_by_layer[layer] = n_by_layer.get(layer, 0) + 1
        for coords, closed in _shapes(el):
            pts = [px(a, b) for a, b in coords]
            if len(pts) < 2:
                continue
            if layer == "road":
                w = max(1, int(round(ROAD_WIDTH_M.get(extra, 5) / m_per_px)))
                draw["paved"].line(pts, fill=255, width=w)
                tr = Image.new("L", (W, H), 0)
                ImageDraw.Draw(tr).line(pts, fill=255, width=1)
                # cheap accumulate on a bounding window only
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x0, x1 = max(0, int(min(xs)) - 1), min(W, int(max(xs)) + 2)
                y0, y1 = max(0, int(min(ys)) - 1), min(H, int(max(ys)) + 2)
                if x1 > x0 and y1 > y0:
                    traffic[y0:y1, x0:x1] += np.asarray(tr.crop((x0, y0, x1, y1)), np.float32) / 255 \
                        * ROAD_TRAFFIC.get(extra, 0.1)
            elif layer == "rail":
                if closed and tags.get("landuse") == "railway":
                    draw["paved"].polygon(pts, fill=255); draw["industrial"].polygon(pts, fill=255)
                else:
                    draw["paved"].line(pts, fill=255, width=max(1, int(round(6 / m_per_px))))
            elif layer == "waterline":
                if tags.get("name") and extra == "river":
                    mid = coords[len(coords) // 2]
                    names.append((tags["name"], mid[0], mid[1], "river"))
                w = max(1, int(round((25 if extra == "river" else 6) / m_per_px)))
                draw["water"].line(pts, fill=255, width=w)
            elif closed and len(pts) >= 3:
                draw[layer].polygon(pts, fill=255)
                if layer == "hub":
                    draw["paved"].polygon(pts, fill=255)
                if layer == "hub" and tags.get("name"):
                    names.append((tags["name"], coords[0][0], coords[0][1], tags.get("amenity")))

    A = {k: np.asarray(v, np.float32) / 255 for k, v in layers.items()}
    water = A["water"]
    building = A["building"] * (1 - water)
    paved = np.clip(A["paved"] - building - water, 0, 1)
    paved = np.maximum(paved, 0.35 * A["commercial"] * (1 - building) * (1 - water))  # hard-standing in commercial plots
    taken = np.clip(water + building + paved, 0, 1)
    green = np.clip(A["green"] + 0.6 * A["grass"], 0, 1) * (1 - taken)
    soil = np.clip(1 - taken - green, 0, 1)

    def agg(a):
        return a.reshape(R, sub, C, sub).mean(axis=(1, 3))

    f = {"water": agg(water), "building": agg(building), "paved": agg(paved),
         "green": agg(green), "soil": agg(soil), "sand": agg(A["sand"])}
    f["concrete"] = f["building"] + f["paved"]
    f["built_up"] = np.clip(f["building"] / 0.45, 0, 1)
    road_density = agg(traffic) * sub                     # weighted road pixels per zone row
    land_act = agg(A["commercial"]) + 1.5 * agg(A["hub"]) + 0.8 * agg(A["industrial"])
    raw_act = road_density / (np.percentile(road_density, 95) + 1e-6) + 2.0 * land_act
    f["activity"] = np.clip(raw_act / (np.percentile(raw_act, 97) + 1e-6), 0, 1) * 0.95 + 0.03

    stats = {"features": n_by_layer, "building_cover": float(f["building"].mean()),
             "green_cover": float(f["green"].mean()), "water_cover": float(f["water"].mean())}
    return f, names, stats


def apply_osm(zones, data, area):
    """Overwrite the synthetic land-use columns of `zones` with OSM-derived values."""
    from zones import _neigh_mean, _km
    f, names, stats = osm_to_zone_features(data, area)
    r, c = zones.row.values, zones.col.values
    for k in ["concrete", "green", "water", "soil", "built_up", "activity"]:
        zones[k] = f[k][r, c]
    tot = zones[["concrete", "green", "water", "soil"]].sum(axis=1)
    for k in ["concrete", "green", "water", "soil"]:
        zones[k] = zones[k] / tot
    G = np.zeros((area["rows"], area["cols"])); G[r, c] = zones.green.values
    zones["green_300m"] = _neigh_mean(G)[r, c]

    landmarks = _pick_landmarks(names)
    if landmarks:
        D = np.vstack([_km(zones.lat.values, zones.lon.values, la, lo) for _, la, lo, _ in landmarks])
        zones["near"] = [landmarks[i][0] for i in D.argmin(0)]
        zones["near_km"] = D.min(0)
    else:
        zones["near"] = "unnamed area"; zones["near_km"] = 0.0

    if stats["building_cover"] < 0.03:
        print("  ⚠ OSM building coverage is very low here – buildings may be unmapped; "
              "concrete share and built-up ratio will be underestimated.")
    return zones, landmarks, stats


def _pick_landmarks(names, limit=30):
    """De-duplicate named places; prefer hubs/stations/markets, then neighbourhoods."""
    rank = {"bus_station": 0, "station": 0, "marketplace": 0, "hospital": 1, "university": 1, "college": 1,
            "river": 1, "suburb": 2, "town": 2, "neighbourhood": 3, "quarter": 3, "village": 4, "hamlet": 5}
    seen, out = set(), []
    for n in sorted(names, key=lambda t: rank.get(t[3], 6)):
        key = n[0].strip().lower()
        if key in seen:
            continue
        seen.add(key); out.append(n)
        if len(out) >= limit:
            break
    return out
