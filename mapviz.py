"""Interactive Leaflet maps (via folium): heat map, cause map, hot-spot cards, solutions."""
import base64
import html
import io
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium import plugins
import branca.colormap as cm
from config import CAUSE_COLORS

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
SEV_COLORS = {"Hot": "#e31a1c", "Watch": "#fd8d3c", "Normal": "#2c7fb8"}


def _base_map(center, zoom=13):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI.format("Canvas/World_Dark_Gray_Base"), attr="Esri, HERE, Garmin, OSM contributors",
                     name="Dark (best for heat)", show=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets (OSM)", show=False).add_to(m)
    folium.TileLayer(tiles=ESRI.format("World_Imagery"), attr="Esri, Maxar, Earthstar Geographics",
                     name="Satellite", show=False).add_to(m)
    folium.TileLayer(tiles=ESRI.format("Canvas/World_Light_Gray_Base"), attr="Esri, HERE, Garmin, OSM contributors",
                     name="Light", show=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)
    plugins.MousePosition(position="bottomleft", prefix="Lat/Lon:", num_digits=4).add_to(m)
    plugins.LocateControl(position="topleft", strings={"title": "Show my location"}).add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="meters").add_to(m)
    return m


HEAT_CMAP = "turbo"


def _raster_uri(grid, vmin, vmax, cmap=HEAT_CMAP, alpha=0.72):
    """Smooth (bicubic) PNG of a zone grid, as a data URI for an ImageOverlay."""
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic", alpha=alpha, aspect="auto")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _hour_grids(typ, z):
    R, C = z.row.max() + 1, z.col.max() + 1
    t = typ.merge(z[["zone_id", "row", "col"]], on="zone_id")
    out = []
    for h in range(24):
        g = np.full((R, C), np.nan)
        th = t[t.hour == h]
        g[th.row.values, th.col.values] = th.temp.values
        out.append(g)
    return out


def _cell_bounds(r, dlat, dlon):
    return [[r.lat - dlat / 2, r.lon - dlon / 2], [r.lat + dlat / 2, r.lon + dlon / 2]]


def _card(r):
    e = html.escape
    sev = SEV_COLORS.get(r.severity, "#999")
    cc = CAUSE_COLORS.get(r.cause, "#999")
    path = "".join(f"<li>{e(s)}</li>" for s in r.decision_path) or "<li>—</li>"
    acts = "".join(f"<li>{e(a)}</li>" for a in r.actions) or "<li>No action needed</li>"
    return f"""
<div style="font-family:system-ui,Segoe UI,Arial;width:300px;font-size:12.5px;line-height:1.4">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <b style="font-size:14px">{e(r.zone_id)}</b>
    <span style="background:{sev};color:#fff;border-radius:10px;padding:1px 9px;font-weight:600">{e(r.severity)} · {e(r.priority)}</span>
  </div>
  <div style="color:#666;margin-bottom:6px">near {e(r.near)} · {r.lat:.4f}, {r.lon:.4f}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;background:#f6f6f6;border-radius:6px;padding:6px;margin-bottom:6px">
    <div>Peak ΔT<br><b style="font-size:15px;color:{sev}">+{r.peak_dT:.1f} °C</b></div>
    <div>Peak temp<br><b style="font-size:15px">{r.peak_temp:.1f} °C</b> <span style="color:#777">@ {int(r.peak_hour):02d}:00</span></div>
    <div style="grid-column:1/3">Hot hours: <b>{e(r.hot_hours)}</b></div>
  </div>
  <div style="border-left:4px solid {cc};padding-left:7px;margin-bottom:6px">
    <div style="color:#777;font-size:11px;text-transform:uppercase;letter-spacing:.04em">Why is it hot? (EXPLAIN)</div>
    <b style="color:{cc};font-size:14px">{e(r.cause_detail)}</b> <span style="color:#777">· vote share {r.confidence:.0%}</span><br>
    <span style="color:#444">{e(r.evidence)}</span>
  </div>
  <details style="margin-bottom:6px"><summary style="cursor:pointer;color:#555">Decision-tree path</summary>
    <ol style="margin:4px 0 0 16px;padding:0;color:#444">{path}</ol></details>
  <div style="color:#777;font-size:11px;text-transform:uppercase;letter-spacing:.04em">What should change? (PRESCRIBE)</div>
  <b>{e(r.headline)}</b>
  <ul style="margin:3px 0 4px 16px;padding:0">{acts}</ul>
  <div style="color:#2e7d32">Expected: {e(r.cooling)}</div>
</div>"""


def _panel(z, area_name, metrics, sources):
    hot, watch = (z.severity == "Hot").sum(), (z.severity == "Watch").sum()
    top = z.sort_values("priority_rank").head(5)
    rows = "".join(
        f"<tr><td style='padding:1px 4px'>{i+1}</td><td>{html.escape(r.near)}</td>"
        f"<td style='color:#ff6b5b;text-align:right'>+{r.peak_dT:.1f}°</td>"
        f"<td><span style='color:{CAUSE_COLORS.get(r.cause)}'>■</span> {html.escape(r.cause)}</td></tr>"
        for i, r in enumerate(top.itertuples()))
    legend = "".join(f"<div><span style='display:inline-block;width:11px;height:11px;background:{c};"
                     f"border-radius:2px;margin-right:6px'></span>{k}</div>"
                     for k, c in CAUSE_COLORS.items() if k != "None")
    src = (f"Land use: {sources['landuse']}. Weather: {sources['weather']}. "
           f"Zone-level sensor effects simulated (TRL 3)")
    return f"""
<div id="uhi-panel" style="position:fixed;top:12px;left:56px;z-index:9999;width:330px;
  background:rgba(18,20,26,.92);color:#eee;border-radius:12px;padding:12px 14px;
  font-family:system-ui,Segoe UI,Arial;font-size:12px;box-shadow:0 4px 18px rgba(0,0,0,.4)">
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <div style="font-size:16px;font-weight:700">Urban Heat Anomaly Map</div>
    <a href="#" onclick="var b=document.getElementById('uhi-body');b.style.display=b.style.display=='none'?'block':'none';return false;"
       style="color:#aaa;text-decoration:none">▾</a></div>
  <div style="color:#ff9d5c;letter-spacing:.08em;font-size:10.5px;margin-bottom:6px">DETECT → EXPLAIN → PRESCRIBE · {html.escape(area_name)}</div>
  <div id="uhi-body">
  <div style="display:flex;gap:6px;margin-bottom:8px">
    <div style="flex:1;background:#2a1414;border-radius:8px;padding:6px;text-align:center"><b style="font-size:18px;color:#ff4d4d">{hot}</b><br>Hot zones</div>
    <div style="flex:1;background:#2a2014;border-radius:8px;padding:6px;text-align:center"><b style="font-size:18px;color:#ffa64d">{watch}</b><br>Watch zones</div>
    <div style="flex:1;background:#142a1a;border-radius:8px;padding:6px;text-align:center"><b style="font-size:18px;color:#5fd38d">{int(z.plant_candidate.sum())}</b><br>Plant sites</div>
  </div>
  <div style="color:#aaa;margin-bottom:2px">Top-5 priority hot spots</div>
  <table style="width:100%;border-collapse:collapse;margin-bottom:8px">{rows}</table>
  <div style="color:#aaa;margin-bottom:2px">Cause colours</div>{legend}
  <div style="color:#888;margin-top:8px;font-size:10.5px">Model: Decision Tree, test accuracy {metrics['accuracy']:.0%} (zone-held-out)<br>
  {html.escape(src)}.<br>Click any coloured zone or 🔥 marker for its diagnosis.</div>
  </div>
</div>"""


def _slider_panel():
    return """
<div style="position:fixed;bottom:28px;left:50%;transform:translateX(-50%);z-index:9999;width:min(560px,92vw);
  background:rgba(18,20,26,.92);color:#eee;border-radius:12px;padding:10px 14px;font:12px system-ui,Segoe UI,Arial;
  box-shadow:0 4px 18px rgba(0,0,0,.4)">
  <div style="display:flex;align-items:center;gap:10px">
    <button id="uhi-play" style="background:#ff7a3d;color:#111;border:0;border-radius:8px;padding:5px 10px;font-weight:700;cursor:pointer">▶ Play 24 h</button>
    <div style="font-size:20px;font-weight:700;min-width:62px" id="uhi-hour-lab">15:00</div>
    <input id="uhi-hour" type="range" min="0" max="23" step="1" value="15" style="flex:1;accent-color:#ff7a3d">
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-top:6px">
    <span id="uhi-lo" style="min-width:58px;text-align:right">–</span>
    <div style="flex:1;height:10px;border-radius:5px;background:linear-gradient(90deg,#30123b,#4686fb,#1ae4b6,#a2fc3c,#fabb39,#e4460a,#7a0403)"></div>
    <span id="uhi-hi" style="min-width:58px">–</span>
  </div>
  <div style="color:#aaa;margin-top:3px;text-align:center">Typical-day air temperature · city median <b id="uhi-med" style="color:#fff">–</b> · colour scale rescales each hour to show where it is hottest</div>
</div>"""


def build_map(zones_df, typ, z, metrics, area_name, out_path, landmarks=(), river=None, sources=None):
    sources = sources or {"landuse": "synthetic", "weather": "simulated climate"}
    dlat, dlon = zones_df.attrs["dlat"], zones_df.attrs["dlon"]
    center = [z.lat.mean(), z.lon.mean()]
    m = _base_map(center)

    # 1) Smooth temperature heat map (raster overlay) with a 24-hour slider
    bounds = [[z.lat.min() - dlat / 2, z.lon.min() - dlon / 2], [z.lat.max() + dlat / 2, z.lon.max() + dlon / 2]]
    grids = _hour_grids(typ, z)
    ranges = [(float(np.nanpercentile(g, 1)), float(np.nanmax(g))) for g in grids]
    uris = [_raster_uri(g, lo, hi) for g, (lo, hi) in zip(grids, ranges)]
    medians = [float(np.nanmedian(g)) for g in grids]
    fg_heat = folium.FeatureGroup(name="🔥 Temperature heat map (use hour slider)", show=True)
    overlay = folium.raster_layers.ImageOverlay(uris[15], bounds=bounds, opacity=1.0, interactive=False, zindex=2)
    overlay.add_to(fg_heat)
    fg_heat.add_to(m)

    # soft glow on anomalous zones only (leaflet.heat), off by default
    fg_glow = folium.FeatureGroup(name="✨ Anomaly glow (ΔT-weighted)", show=False)
    hotpts = [[r.lat, r.lon, float(min(r.peak_dT / 3.0, 1.0))] for r in z[z.peak_dT > 0.8].itertuples()]
    plugins.HeatMap(hotpts, radius=18, blur=14, min_opacity=0.3,
                    gradient={0.3: "#fee090", 0.6: "#f46d43", 1.0: "#a50026"}).add_to(fg_glow)
    fg_glow.add_to(m)

    # 2) Zone grid coloured by anomaly dT
    cmap = cm.LinearColormap(["#2166ac", "#67a9cf", "#f7f7f7", "#fdae61", "#d7191c", "#67000d"],
                             vmin=-1.5, vmax=3.0, caption="Peak anomaly ΔT vs city median (°C)")
    fg_grid = folium.FeatureGroup(name="▦ Anomaly grid (ΔT per zone)", show=False)
    for r in z.itertuples():
        folium.Rectangle(_cell_bounds(r, dlat, dlon), weight=0, fill=True, fill_opacity=0.55,
                         fill_color=cmap(np.clip(r.peak_dT, -1.5, 3.0)),
                         tooltip=f"{r.zone_id} · ΔT {r.peak_dT:+.1f} °C · {r.severity} · {r.near}").add_to(fg_grid)
    fg_grid.add_to(m)

    # 3) Diagnosed causes (Watch + Hot zones) with full diagnosis popup
    fg_cause = folium.FeatureGroup(name="◼ Causes of heat (Watch/Hot zones)", show=True)
    for r in z[z.severity != "Normal"].itertuples():
        folium.Rectangle(_cell_bounds(r, dlat, dlon),
                         color=CAUSE_COLORS[r.cause], weight=1.2 if r.severity == "Watch" else 2.5,
                         fill=True, fill_color=CAUSE_COLORS[r.cause],
                         fill_opacity=0.35 if r.severity == "Watch" else 0.6,
                         tooltip=f"{r.cause_detail} · ΔT +{r.peak_dT:.1f} °C · click for fix",
                         popup=folium.Popup(_card(r), max_width=330)).add_to(fg_cause)
    fg_cause.add_to(m)

    # 4) Priority hot-spot markers
    fg_p1 = folium.FeatureGroup(name="📍 Worst hot spot in each area", show=True)
    worst = (z[z.priority.isin(["P1", "P2"])].sort_values("priority_rank")
             .drop_duplicates("near").head(10))
    for r in worst.itertuples():
        folium.Marker([r.lat, r.lon],
                      icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                      tooltip=f"#{r.priority_rank} {r.near} · +{r.peak_dT:.1f} °C · {r.cause_detail}",
                      popup=folium.Popup(_card(r), max_width=330)).add_to(fg_p1)
    fg_p1.add_to(m)

    # 5) Solutions: plantation candidates & cool refuges
    fg_tree = folium.FeatureGroup(name="🌳 Tree-planting candidate points", show=True)
    for r in z[z.plant_candidate].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=6, color="#1b5e20", weight=2, fill=True,
                            fill_color="#66bb6a", fill_opacity=0.9,
                            tooltip=f"Plant here · {r.zone_id} · soil {r.soil:.0%}, green ≤300 m {r.green_300m:.0%}").add_to(fg_tree)
    fg_tree.add_to(m)
    fg_cool = folium.FeatureGroup(name="💧 Cool refuges (heat shelters)", show=False)
    for r in z[z.cool_refuge].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=7, color="#0d47a1", weight=2, fill=True,
                            fill_color="#4fc3f7", fill_opacity=0.85,
                            tooltip=f"Cool refuge · mean ΔT {r.mean_dT:+.1f} °C · {r.near}").add_to(fg_cool)
    fg_cool.add_to(m)

    # 6) Landmarks + river
    real = sources["landuse"].startswith("OpenStreetMap")
    fg_lm = folium.FeatureGroup(name="🏷 Place names" + ("" if real else " (approx.)"), show=True)
    if river:
        folium.PolyLine(river["points"], color="#c2a36b", weight=6, opacity=0.6,
                        tooltip=river["label"]).add_to(fg_lm)
    if True:
        for name, la, lo, *_ in landmarks:
            folium.Marker([la, lo], icon=folium.DivIcon(html=
                f"<div style='font:600 11px system-ui;color:#fff;text-shadow:0 0 3px #000,0 0 3px #000;"
                f"white-space:nowrap;transform:translate(-50%,-50%)'>{html.escape(name)}</div>")).add_to(fg_lm)
    fg_lm.add_to(m)

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.get_root().html.add_child(folium.Element(_panel(z, area_name, metrics, sources)))
    m.get_root().html.add_child(folium.Element(_slider_panel()))
    m.get_root().script.add_child(folium.Element(
        f"""
    window.addEventListener('load', function(){{
      var URIS = {json.dumps(uris)};
      var RNG = {json.dumps([[round(a, 1), round(b, 1)] for a, b in ranges])};
      var MED = {json.dumps([round(v, 1) for v in medians])};
      var ov = {overlay.get_name()};
      var sl = document.getElementById('uhi-hour'), lab = document.getElementById('uhi-hour-lab');
      var lo = document.getElementById('uhi-lo'), hi = document.getElementById('uhi-hi'), md = document.getElementById('uhi-med');
      var timer = null, btn = document.getElementById('uhi-play');
      function show(h) {{
        ov.setUrl(URIS[h]); sl.value = h;
        lab.textContent = (h < 10 ? '0' : '') + h + ':00';
        lo.textContent = RNG[h][0].toFixed(1) + ' °C'; hi.textContent = RNG[h][1].toFixed(1) + ' °C';
        md.textContent = MED[h].toFixed(1) + ' °C';
      }}
      sl.addEventListener('input', function(){{ show(+sl.value); }});
      btn.addEventListener('click', function(){{
        if (timer) {{ clearInterval(timer); timer = null; btn.textContent = '▶ Play 24 h'; return; }}
        btn.textContent = '❚❚ Pause';
        timer = setInterval(function(){{ show((+sl.value + 1) % 24); }}, 700);
      }});
      show(15);
    }});
    """))
    m.fit_bounds([[z.lat.min(), z.lon.min()], [z.lat.max(), z.lon.max()]])
    m.save(out_path)
    return out_path
