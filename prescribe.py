"""PRESCRIBE: map cause + local context -> targeted interventions and a priority score.

Cooling figures are indicative ranges for planning discussion only; they must be
validated locally (next TRL)."""
import numpy as np
import pandas as pd

ACTIONS = {
    "Surface retention": {
        "headline": "Cool the surfaces",
        "actions": ["Cool-roof / high-albedo coating on flat RCC roofs",
                    "Reflective or permeable paving for parking lots & lanes",
                    "Shade sails / pergolas over bus bays and market walkways"],
        "cooling": "≈0.5–1.5 °C surface-air reduction (evening peak)",
    },
    "Low ventilation": {
        "headline": "Let the air move",
        "actions": ["Keep a wind corridor open along main roads (no new blocking frontage)",
                    "Building setbacks / open ground floors in new approvals",
                    "Misting points + shaded rest spots during stagnant hours"],
        "cooling": "≈0.3–1.0 °C via restored airflow",
    },
    "Activity spike": {
        "headline": "Shift the heat source",
        "actions": ["Heavy-vehicle / loading time-window outside 2–6 PM",
                    "Signal re-timing to cut idling at congested junctions",
                    "Shaded e-rickshaw / bus bays; promote EV feeder routes"],
        "cooling": "≈0.5–2.0 °C at peak hours (traffic/commerce heat)",
    },
    "Low green cover": {
        "headline": "Add shade & green",
        "actions": ["Street-tree plantation at candidate points (native: neem, pungai, vaagai)",
                    "Pocket parks / Miyawaki plots on vacant soil patches",
                    "Green roofs or creepers on public buildings"],
        "cooling": "≈1–2 °C daytime under mature canopy",
    },
    "Mixed": {
        "headline": "Combined package",
        "actions": ["Pair cool surfaces with tree shade on the same street segment",
                    "Re-time heavy activity + add shaded pedestrian route",
                    "Install a local sensor to separate drivers before major spend"],
        "cooling": "Depends on dominant driver – monitor first",
    },
}


def evidence(row):
    bits = []
    if row.concrete >= 0.55: bits.append(f"concrete {row.concrete:.0%}")
    if row.peak_wind <= 1.5: bits.append(f"wind {row.peak_wind:.1f} m/s")
    if row.activity >= 0.6: bits.append(f"activity index {row.activity:.2f}")
    if row.green_300m <= 0.15: bits.append(f"green ≤300 m {row.green_300m:.0%}")
    if row.soil >= 0.4: bits.append(f"bare soil {row.soil:.0%}")
    return " • ".join(bits) or "no single extreme feature"


def prescribe(z):
    z = z.copy()
    hot = z.severity != "Normal"
    z["evidence"] = z.apply(evidence, axis=1)
    z["headline"] = z.cause.map(lambda c: ACTIONS.get(c, {}).get("headline", ""))
    z["actions"] = z.cause.map(lambda c: ACTIONS.get(c, {}).get("actions", []))
    z["cooling"] = z.cause.map(lambda c: ACTIONS.get(c, {}).get("cooling", ""))
    # exposure proxy = how many people are likely there (built-up + activity)
    exposure = 0.6 * z.built_up + 0.4 * z.activity
    z["priority_score"] = np.where(hot, z.peak_dT.clip(lower=0) * (0.5 + exposure) * (0.5 + 0.5 * z.confidence), 0)
    z["priority_rank"] = z.priority_score.rank(ascending=False, method="first").astype(int)
    z["priority"] = np.select([hot & (z.priority_rank <= 15), hot & (z.priority_rank <= 50), hot],
                              ["P1", "P2", "P3"], "—")
    # Mixed zones: combine the leading actions of the two strongest drivers
    if "cause_detail" in z:
        for i in z.index[(z.cause == "Mixed") & hot]:
            parts = [p.strip() for p in z.at[i, "cause_detail"].replace("Mixed:", "").split("+")]
            parts = [p for p in parts if p in ACTIONS]
            if len(parts) == 2:
                z.at[i, "actions"] = ACTIONS[parts[0]]["actions"][:2] + ACTIONS[parts[1]]["actions"][:2]
                z.at[i, "headline"] = f"{ACTIONS[parts[0]]['headline']} + {ACTIONS[parts[1]]['headline'].lower()}"
                z.at[i, "cooling"] = "Combined effect of both measures – validate with a local sensor"
    # local special case: dry riverbed -> plant the banks, not the channel
    river = z.near.str.contains("river", case=False) & (z.near_km < 0.5) & hot
    z.loc[river, "headline"] = "Green the riverbanks"
    z.loc[river, "actions"] = pd.Series([["Riparian tree buffer along both banks (not inside the channel)",
                                          "Check sand mining / exposed-sand patches",
                                          "Shaded walkway + trees on bridge approaches"]] * int(river.sum()),
                                        index=z.index[river], dtype=object)
    # plantation candidates: heat-affected zones that still have open soil to plant on
    z["plant_candidate"] = hot & z.cause.isin(["Low green cover", "Surface retention", "Mixed"]) \
        & (z.soil >= 0.20) & (z.green_300m <= 0.25) \
        & ~(z.near.str.contains("river", case=False) & (z.near_km < 0.2)) & (z.water < 0.3)
    # cool refuges: coolest zones (shade / water) usable as heat shelters
    z["cool_refuge"] = z.mean_dT <= z.mean_dT.quantile(0.04)
    return z
