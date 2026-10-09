"""EXPLAIN: interpretable Decision Tree that attributes each anomaly to a dominant cause."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from config import WATCH_DT
from simulate import ground_truth_cause

FEATURES = ["temp", "rh", "wind", "concrete", "green_300m", "built_up", "activity", "hour"]
FEATURE_LABELS = {"temp": "Temperature (°C)", "rh": "Humidity (%)", "wind": "Wind (m/s)",
                  "concrete": "Concrete share", "green_300m": "Green cover ≤300 m",
                  "built_up": "Built-up ratio", "activity": "Activity index", "hour": "Hour"}


def train(df, max_depth=6, seed=0):
    anom = df[df.dT >= WATCH_DT].copy()
    anom["cause_true"] = ground_truth_cause(anom)
    X, y = anom[FEATURES], anom.cause_true
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)  # split BY ZONE
    tr, te = next(split.split(X, y, groups=anom.zone_id))
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=40,
                                 class_weight="balanced", random_state=seed)
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])
    metrics = {"n_anomalous_records": len(anom), "n_train": len(tr), "n_test": len(te),
               "accuracy": accuracy_score(y.iloc[te], pred),
               "report": classification_report(y.iloc[te], pred, zero_division=0),
               "rules": export_text(clf, feature_names=FEATURES, decimals=2)}
    return clf, anom, metrics


def decision_path_text(clf, x_row):
    """Human-readable branch path for one record, e.g. 'Concrete share 0.78 > 0.55 → ...'."""
    X = x_row[FEATURES].to_frame().T.astype(float)
    node_ids = clf.decision_path(X).indices
    t = clf.tree_
    steps = []
    for n in node_ids:
        if t.children_left[n] == t.children_right[n]:
            continue
        f = FEATURES[t.feature[n]]
        v, thr = X.iloc[0][f], t.threshold[n]
        op = "≤" if v <= thr else ">"
        steps.append(f"{FEATURE_LABELS[f]} {v:.2f} {op} {thr:.2f}")
    return steps


def diagnose_zones(clf, df, zones_summary):
    """Per zone: classify every anomalous hour, vote (weighted by dT) -> cause + confidence."""
    anom = df[df.dT >= WATCH_DT].copy()
    anom["pred"] = clf.predict(anom[FEATURES])
    votes = anom.groupby(["zone_id", "pred"]).dT.sum().reset_index()
    tot = votes.groupby("zone_id").dT.transform("sum")
    votes["share"] = votes.dT / tot
    best = votes.loc[votes.groupby("zone_id").share.idxmax(), ["zone_id", "pred", "share"]]
    best.columns = ["zone_id", "cause", "confidence"]
    z = zones_summary.merge(best, on="zone_id", how="left")
    z["cause"] = z.cause.fillna("None")
    z.loc[z.severity == "Normal", "cause"] = "None"
    z["confidence"] = z.confidence.fillna(0.0)
    # for Mixed zones, name the two strongest single drivers among the votes
    single = votes[votes.pred != "Mixed"].sort_values("share", ascending=False)
    top2 = single.groupby("zone_id").pred.apply(lambda s: " + ".join(s.iloc[:2]))
    z["cause_detail"] = np.where(z.cause == "Mixed", "Mixed: " + z.zone_id.map(top2).fillna("several drivers"), z.cause)

    # decision path from the zone's most anomalous typical hour (median-day record)
    paths = {}
    for zid, ph in z.loc[z.severity != "Normal", ["zone_id", "peak_hour"]].values:
        rows = df[(df.zone_id == zid) & (df.hour == ph)]
        rec = rows.iloc[(rows.dT - rows.dT.median()).abs().argmin()]
        paths[zid] = decision_path_text(clf, rec)
    z["decision_path"] = z.zone_id.map(lambda k: paths.get(k, []))
    return z
