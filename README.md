# Urban Heat Anomaly Diagnosis & Action
**DETECT → EXPLAIN → PRESCRIBE** — a zone-level smart-city prototype (TRL 3) for the Vellore–Katpadi area.

Most urban-heat-island (UHI) tools stop at *"where is it hot?"*. This project goes on to *"why is it hot here?"* and *"what should change?"* for every ~300 m zone, and shows the answer on an interactive map.

## Run it
```bash
pip install -r requirements.txt
python main.py                      # Vellore–Katpadi demo, synthetic land use, ~10 s
python main.py --osm --live         # REAL land use (OpenStreetMap) + REAL weather (Open-Meteo) – recommended
python main.py --osm                # real land use only
python main.py --live               # real weather only
python main.py --lat 13.0827 --lon 80.2707 --name "Chennai Central" --span-km 8   # any other area
```
Then open **`output/heat_map.html`** in a browser (it needs internet for the map tiles).

## APIs used (both free, no key, no sign-up)
| API | Flag | What it provides |
|---|---|---|
| **OpenStreetMap Overpass** | `--osm` | Buildings, roads (by class), parking, rail yards, parks, forests, trees, grass, water, rivers, sand, commercial/industrial areas, bus stands, markets and named places for the study area. These become each zone's concrete %, green %, water %, built-up ratio, activity index and place name. |
| **Open-Meteo** | `--live` | Real hourly temperature, humidity and wind for the past 7 days, used as the regional baseline. |

* The first `--osm` run queries Overpass (30–90 s) and saves the result in `osm_cache/`. Later runs reuse it, so **your demo works offline** after one successful run. Use `--refresh-osm` to download fresh data.
* If Overpass is busy it tries two mirror servers; if all fail, the run continues with synthetic land use and prints a warning.
* `--osm-file my_export.json` loads a saved Overpass JSON instead of querying.
* **OSM caveat:** OSM coverage in Indian cities is uneven — many buildings and trees are unmapped. The program warns when building coverage looks too low; the green-cover threshold adapts to the area so sparse mapping doesn't flag every zone.

## Map shows only the panels / blank page?
Map libraries are now **bundled inside `heat_map.html`** (from `./vendor`), because some Indian ISPs block `cdn.jsdelivr.net`. Only the background map tiles (OpenStreetMap / Esri) need internet. If even the tiles are blocked, the heat colours and zones still draw on a grey background. Use `--cdn` to go back to CDN loading.

## What you get (`output/`)
| File | What it shows |
|---|---|
| `heat_map.html` | Interactive map: smooth temperature heat map with a **24-hour slider / play button**, cause-coloured hot zones, 🔥 worst spot per area, 🌳 tree-planting points, 💧 cool refuges, dark/street/satellite base maps. Click any zone for its full diagnosis card. |
| `dashboard.png` | 6-panel summary for slides/report: 3 PM heat map, severity, cause map, cause-by-hour curves, model drivers, top actions. |
| `decision_tree.png` | First levels of the cause classifier (explainability). |
| `zone_report.csv` | All 784 zones: ΔT, severity, hot hours, cause, evidence, actions, priority. |
| `action_plan.md` | Ranked, human-readable plan: causes found, area summary table, P1/P2 hot spots with fixes. |
| `model_rules.txt` | Full tree rules + precision/recall per cause. |

## How it works (maps to the slides)
| Stage | File | Method |
|---|---|---|
| Zones | `zones.py`, `osm_landuse.py` | 28×28 grid (~300 m). With `--osm`: OSM features rasterised at ~10 m and aggregated per zone (concrete = buildings + paved, green = trees/parks + 0.6×grass, activity = traffic-weighted road length + commercial/market/transit area). Without it: synthetic land use seeded from approximate landmarks. |
| Sensor streams | `simulate.py` | 14 days × 24 h per zone. Temperature = diurnal baseline + **surface retention** (concrete releases heat in the evening) + **low ventilation** (built-up blocks wind) + **activity spike** (2–6 PM traffic/commerce) + **low green cover** (≤300 m) − water/vegetation cooling + noise. |
| DETECT | `detect.py` | ΔT = zone temp − city median at the same hour. Normal < 0.8 °C ≤ Watch < 1.8 °C ≤ Hot. |
| EXPLAIN | `attribute.py` | `DecisionTreeClassifier` (depth 6) on temp, RH, wind, concrete, green ≤300 m, built-up, activity, hour. Split **by zone** for honest testing (~80 % accuracy). Each zone's cause = ΔT-weighted vote over its anomalous hours + the decision path. |
| PRESCRIBE | `prescribe.py` | Cause → intervention package, priority = ΔT × exposure × confidence (P1/P2/P3), plantation candidates (hot + open soil + low green), cool refuges. Local rule: riverbed → green the banks, not the channel. |
| Visualise | `mapviz.py`, `dashboard.py` | Folium/Leaflet map + Matplotlib dashboard. |

## Honest limits (say this in the viva)
* With `--osm --live`, land use and the weather baseline are **real**; the per-zone temperature response is still **simulated** from them (TRL 3) — that is what gives ground-truth cause labels to train and test the classifier.
* Without `--osm`, landmark positions are approximate and land-use fractions are synthetic.
* Cooling figures in the actions are indicative planning ranges, not measured results.

## Upgrade path to real data (next TRL)
1. **Land surface temperature:** Landsat 8/9 Band 10 or MODIS LST via Google Earth Engine → replace the simulated `temp`.
2. **Green cover:** Sentinel-2 NDVI to fill the trees and green space OSM hasn't mapped.
3. **Ground truth:** a few low-cost ESP32 + DHT22/BME280 nodes in 3–4 zones to validate ΔT and causes.
4. **Model:** compare Random Forest / Gradient Boosting, add SHAP explanations.
