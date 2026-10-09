"""
Urban Heat Anomaly Diagnosis & Action — end-to-end pipeline.

    python main.py                        # Vellore–Katpadi demo (simulated)
    python main.py --live                 # use real regional weather from Open-Meteo
    python main.py --lat 13.0827 --lon 80.2707 --name "Chennai Central" --span-km 8

Outputs (in ./output):
    heat_map.html        interactive map: 24-hour heat map, causes, hot spots, solutions
    dashboard.png        static summary dashboard for slides / report
    decision_tree.png    top levels of the cause classifier
    zone_report.csv      every zone: dT, severity, cause, evidence, actions, priority
    action_plan.md       ranked, human-readable action plan
    model_rules.txt      full decision-tree rules + evaluation
"""
import argparse
import os
import time
import numpy as np
from config import AREA, LANDMARKS, RIVER
import zones as Z
import osm_landuse as OSM
from simulate import simulate, fetch_live_weather
from detect import add_anomaly, zone_profiles
from attribute import train, diagnose_zones
from prescribe import prescribe
from mapviz import build_map
from offline_assets import inline_assets
from dashboard import build_dashboard, build_tree_figure


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, help="centre latitude of a custom area")
    p.add_argument("--lon", type=float, help="centre longitude of a custom area")
    p.add_argument("--name", default=None, help="name for a custom area")
    p.add_argument("--span-km", type=float, default=8.0, help="width/height of a custom area")
    p.add_argument("--grid", type=int, default=28, help="zones per side")
    p.add_argument("--days", type=int, default=14, help="simulated days")
    p.add_argument("--live", action="store_true", help="fetch real regional weather (Open-Meteo)")
    p.add_argument("--osm", action="store_true", help="real land use from OpenStreetMap (Overpass API)")
    p.add_argument("--osm-file", help="use a saved Overpass JSON file instead of querying")
    p.add_argument("--refresh-osm", action="store_true", help="ignore the OSM cache and query again")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="output")
    p.add_argument("--cdn", action="store_true", help="load map libraries from CDNs instead of bundling them")
    return p.parse_args()


def write_action_plan(z, metrics, area_name, path, sources):
    hot = z[z.severity != "Normal"].sort_values("priority_rank")
    lines = [f"# Heat Action Plan — {area_name}", "",
             f"- Zones analysed: **{len(z)}**  |  Hot: **{(z.severity=='Hot').sum()}**  |  "
             f"Watch: **{(z.severity=='Watch').sum()}**  |  Tree-planting candidates: **{int(z.plant_candidate.sum())}**",
             f"- Cause classifier: Decision Tree, zone-held-out test accuracy **{metrics['accuracy']:.1%}**",
             f"- Land use: {sources['landuse']}  |  Weather baseline: {sources['weather']}",
             "- Zone-level sensor effects are simulated (TRL 3). Figures are illustrative; validate locally before spending.", "",
             "## Causes found", ""]
    for c, n in hot.cause.value_counts().items():
        lines.append(f"- **{c}** — {n} zones (main areas: {', '.join(hot[hot.cause==c].near.value_counts().index[:3])})")
    lines += ["", "## Area-level summary", "", "| Area | Zones | Worst ΔT | Main cause | Recommended fix |",
              "|---|---|---|---|---|"]
    for near, g in sorted(hot.groupby("near"), key=lambda kv: -kv[1].peak_dT.max()):
        main = g.cause.value_counts().index[0]
        fix = g[g.cause == main].headline.iloc[0]
        lines.append(f"| {near} | {len(g)} | +{g.peak_dT.max():.1f} °C | {main} | {fix} |")
    lines += ["", "## Ranked hot spots (P1 + P2)", ""]
    for r in hot[hot.priority.isin(["P1", "P2"])].itertuples():
        lines += [f"### #{r.priority_rank} · {r.zone_id} near {r.near} ({r.priority})",
                  f"- Location: {r.lat:.4f}, {r.lon:.4f}  ·  peak ΔT **+{r.peak_dT:.1f} °C** at {int(r.peak_hour):02d}:00  ·  hot hours: {r.hot_hours}",
                  f"- **Cause:** {r.cause_detail} (vote share {r.confidence:.0%}) — {r.evidence}",
                  f"- **Decision path:** " + " → ".join(r.decision_path),
                  f"- **Fix — {r.headline}:** " + "; ".join(r.actions),
                  f"- Expected effect: {r.cooling}", ""]
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def main():
    a = parse()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)

    if a.lat is not None and a.lon is not None:
        half_lat = a.span_km / 2 / 111.0
        half_lon = a.span_km / 2 / (111.0 * np.cos(np.radians(a.lat)))
        area = {"name": a.name or f"Area @ {a.lat:.3f}, {a.lon:.3f}",
                "lat_min": a.lat - half_lat, "lat_max": a.lat + half_lat,
                "lon_min": a.lon - half_lon, "lon_max": a.lon + half_lon, "rows": a.grid, "cols": a.grid}
        landmarks, river = Z.random_landmarks(area, seed=a.seed), None
    else:
        area = dict(AREA, rows=a.grid, cols=a.grid)
        landmarks, river = LANDMARKS, RIVER

    print("Urban Heat Anomaly – version 1.3 (offline map libraries)")
    print(f"[1/6] Building {area['rows']}x{area['cols']} zones for {area['name']}")
    zones = Z.build_zones(area, landmarks, river, seed=a.seed)
    landuse_src = "synthetic (seeded from approximate landmarks)"
    map_landmarks = [(n, la, lo, p) for n, la, lo, p, *_ in landmarks]
    if a.osm or a.osm_file:
        print("[1/6] Fetching real land use from OpenStreetMap")
        try:
            data = OSM.load_osm_file(a.osm_file) if a.osm_file else OSM.fetch_osm(area, refresh=a.refresh_osm)
            zones, osm_lm, st = OSM.apply_osm(zones, data, area)
            map_landmarks, river = osm_lm, None
            landuse_src = "OpenStreetMap (Overpass API)"
            print(f"       buildings cover {st['building_cover']:.1%} · green {st['green_cover']:.1%} · "
                  f"water {st['water_cover']:.1%} · {len(osm_lm)} named places")
        except Exception as e:
            print(f"  [OSM unavailable: {e}] -> using synthetic land use")

    live = None
    if a.live:
        print("[2/6] Fetching live regional weather (Open-Meteo)…")
        live = fetch_live_weather((area["lat_min"] + area["lat_max"]) / 2, (area["lon_min"] + area["lon_max"]) / 2)
    print("[2/6] Simulating zone sensor streams")
    df = simulate(zones, days=a.days, seed=a.seed + 4, live=live)

    print("[3/6] DETECT  – anomalies vs city median")
    df = add_anomaly(df)
    typ, zs = zone_profiles(df)
    zs = zs.merge(zones, on="zone_id")

    print("[4/6] EXPLAIN – training Decision Tree cause classifier")
    clf, anom, metrics = train(df)
    print(f"       {metrics['n_anomalous_records']:,} anomalous records · test accuracy {metrics['accuracy']:.1%}")
    z = diagnose_zones(clf, df, zs)

    print("[5/6] PRESCRIBE – interventions + priority")
    z = prescribe(z)

    print("[6/6] Writing maps, dashboard and reports")
    sources = {"landuse": landuse_src,
               "weather": "Open-Meteo (live)" if live is not None else "simulated climate"}
    build_map(zones, typ, z, metrics, area["name"], os.path.join(a.out, "heat_map.html"),
              landmarks=map_landmarks, river=river, sources=sources)
    if not a.cdn:
        inline_assets(os.path.join(a.out, "heat_map.html"))
    build_dashboard(z, typ, clf, metrics, area["name"], os.path.join(a.out, "dashboard.png"),
                    landmarks=map_landmarks, subtitle=f"Land use: {landuse_src} · Weather: {sources['weather']}")
    build_tree_figure(clf, os.path.join(a.out, "decision_tree.png"))
    cols = ["priority_rank", "priority", "zone_id", "near", "lat", "lon", "severity", "peak_dT", "peak_temp",
            "peak_hour", "hot_hours", "cause", "cause_detail", "confidence", "evidence", "headline", "actions", "cooling",
            "concrete", "green_300m", "built_up", "activity", "peak_wind", "plant_candidate", "cool_refuge"]
    out = z.sort_values("priority_rank")[cols].copy()
    out["actions"] = out.actions.map(" | ".join)
    out.round(3).to_csv(os.path.join(a.out, "zone_report.csv"), index=False)
    write_action_plan(z, metrics, area["name"], os.path.join(a.out, "action_plan.md"), sources)
    with open(os.path.join(a.out, "model_rules.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy (zone-held-out): {metrics['accuracy']:.3f}\n\n{metrics['report']}\n\n{metrics['rules']}")

    print(f"\nDone in {time.time()-t0:.1f}s → open {a.out}/heat_map.html in a browser")
    print(z[z.severity != "Normal"].cause.value_counts().to_string())


if __name__ == "__main__":
    main()
