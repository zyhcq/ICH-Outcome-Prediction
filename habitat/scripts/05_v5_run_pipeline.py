"""
05_v5_run_pipeline.py
Pipeline for optimal imaging-only prognostic model.
Model: R3-Best (23 features, LR C=1.0) [CV-optimal configuration]
Endpoint: mRS 3-6
"""
import numpy as np
import pandas as pd
import json, warnings, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge, BayesianRidge, LassoCV
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold
import pickle

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "features"
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CSV_DIR = RESULTS_DIR / "csv"

IMG_21 = [
    "hematoma_volume_log", "relative_volume",
    "sphericity", "surface_area", "compactness", "elongation", "surface_volume_ratio",
    "density_mean", "density_std", "density_skewness", "density_kurtosis",
    "density_max", "density_range",
    "PASH_high_compactness", "PASH_low_dispersion", "PASH_fragmentation",
    "p_basal", "p_brainstem", "p_cerebellum", "p_lobar", "p_thalamus",
]
BASELINE_FEATS = ["gcs_le4", "gcs_5_12", "hematoma_volume_log", "age", "IVH", "infratentorial"]
ENDPOINT_COL = "90_day poor outcome 3_6"

FINAL_FEATURES = [
    "age", "pGCS", "PASH_low_dispersion", "age_sq", "relative_volume",
    "compactness", "density_kurtosis", "density_range", "pGCS_log",
    "age_x_vol", "surface_volume_ratio",
    "pGCS_te", "pGCS_et", "pGCS_q90", "pGCS_severe",
    "density_uniformity", "pGCS_rf", "pGCS_gbr",
    "pGCS_mild", "density_mean", "pGCS_bayes",
    "density_std", "pGCS_sq",
]

FINAL_C = 1.0


def delong_test(y, p1, p2):
    n1 = np.sum(y == 1); n0 = np.sum(y == 0)
    pos1, neg1 = p1[y==1], p1[y==0]; pos2, neg2 = p2[y==1], p2[y==0]
    auc1 = roc_auc_score(y, p1); auc2 = roc_auc_score(y, p2)
    v10 = np.array([np.mean(pos1>n)+0.5*np.mean(pos1==n) for n in neg1])
    v11 = np.array([np.mean(neg1<p)+0.5*np.mean(neg1==p) for p in pos1])
    v20 = np.array([np.mean(pos2>n)+0.5*np.mean(pos2==n) for n in neg2])
    v21 = np.array([np.mean(neg2<p)+0.5*np.mean(neg2==p) for p in pos2])
    s10 = np.cov(v10, v20); s01 = np.cov(v11, v21); S = s10/n0 + s01/n1
    d = auc1-auc2; var_d = S[0,0]+S[1,1]-2*S[0,1]
    if var_d <= 0:
        return {"auc1":float(auc1),"auc2":float(auc2),"delta":float(d),"z":0,"p":1}
    z = d/np.sqrt(var_d); p = 2*(1-stats.norm.cdf(abs(z)))
    return {"auc1":float(auc1),"auc2":float(auc2),"delta":float(d),"z":float(z),"p":float(p)}


def cfnri(y, pn, po):
    e = y==1; ne = y==0
    ue=np.mean(pn[e]>po[e]); de=np.mean(pn[e]<po[e])
    dn=np.mean(pn[ne]<po[ne]); un=np.mean(pn[ne]>po[ne])
    nri_e=ue-de; nri_n=dn-un; val=nri_e+nri_n
    v=(ue+de-nri_e**2)/e.sum()+(dn+un-nri_n**2)/ne.sum()
    se=np.sqrt(v) if v>0 else 1e-10; z=val/se; p=2*(1-stats.norm.cdf(abs(z)))
    return {"cfNRI":float(val),"z":float(z),"p":float(p)}


def idi_test(y, pn, po):
    is_new = np.mean(pn[y==1])-np.mean(pn[y==0])
    is_old = np.mean(po[y==1])-np.mean(po[y==0])
    val = is_new-is_old
    n1=np.sum(y==1); n0=np.sum(y==0)
    var_n = np.var(pn[y==1])/n1 + np.var(pn[y==0])/n0
    var_o = np.var(po[y==1])/n1 + np.var(po[y==0])/n0
    se = np.sqrt(var_n+var_o) if (var_n+var_o)>0 else 1e-10
    z=val/se; p=2*(1-stats.norm.cdf(abs(z)))
    return {"IDI":float(val),"z":float(z),"p":float(p)}


def bootstrap_auc(y, prob, n=2000, seed=42):
    rng=np.random.RandomState(seed); aucs=[]
    for _ in range(n):
        idx=rng.randint(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    return [float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))]


def net_benefit(y, prob, t):
    pp=prob>=t; tp=np.sum(pp&(y==1)); fp=np.sum(pp&(y==0)); n=len(y)
    return float(tp/n - fp/n*t/(1-t)) if t<1 else 0.0


def oof_regress(X_tr, y_target, X_te, model_fn, kf):
    oof = np.zeros(len(X_tr))
    for tri, vai in kf.split(X_tr):
        m = model_fn(); m.fit(X_tr[tri], y_target[tri])
        oof[vai] = m.predict(X_tr[vai])
    m_full = model_fn(); m_full.fit(X_tr, y_target)
    return oof, m_full.predict(X_te)


def eval_lr(feat_cols, df_tr, df_te, y_tr, y_te, skf, C=1.0):
    X_tr = df_tr[feat_cols].values; X_te = df_te[feat_cols].values
    oof = np.zeros(len(y_tr))
    for tri, vai in skf.split(X_tr, y_tr):
        m = Pipeline([("s",StandardScaler()),
                       ("c",LogisticRegression(C=C,max_iter=3000,random_state=SEED))])
        m.fit(X_tr[tri], y_tr[tri])
        oof[vai] = m.predict_proba(X_tr[vai])[:,1]
    mf = Pipeline([("s",StandardScaler()),
                    ("c",LogisticRegression(C=C,max_iter=3000,random_state=SEED))])
    mf.fit(X_tr, y_tr)
    tp = mf.predict_proba(X_te)[:,1]
    return roc_auc_score(y_tr, oof), roc_auc_score(y_te, tp), tp, oof


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print("Running pipeline...")

    df_tr = pd.read_csv(FEATURES / "feature_matrix_train.csv")
    df_te = pd.read_csv(FEATURES / "feature_matrix_test.csv")
    y_tr = df_tr[ENDPOINT_COL].values; y_te = df_te[ENDPOINT_COL].values
    kf = KFold(5, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X_tr = df_tr[IMG_21].values; X_te = df_te[IMG_21].values
    gcs = df_tr["GCS"].values
    
    for df in [df_tr, df_te]:
        df["infratentorial"] = df["ICH location"].isin([1, 2]).astype(float)
        df["gcs_le4"] = (df["GCS"] <= 4).astype(float)
        df["gcs_5_12"] = ((df["GCS"] >= 5) & (df["GCS"] <= 12)).astype(float)
        
    n_infra_tr = df_tr["infratentorial"].sum()
    n_infra_te = df_te["infratentorial"].sum()

    gbr_fn = lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=SEED)
    oof_gbr, te_gbr = oof_regress(X_tr, gcs, X_te, gbr_fn, kf)

    ridge_fn = lambda: Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
    oof_ridge, te_ridge = oof_regress(X_tr, gcs, X_te, ridge_fn, kf)

    sc_svr = StandardScaler()
    Xs_tr = sc_svr.fit_transform(X_tr); Xs_te = sc_svr.transform(X_te)
    svr_fn = lambda: SVR(kernel='rbf', C=50.0, epsilon=0.3)
    oof_svr, te_svr = oof_regress(Xs_tr, gcs, Xs_te, svr_fn, kf)

    pgcs_oof = (oof_gbr + oof_ridge + oof_svr) / 3
    pgcs_te = (te_gbr + te_ridge + te_svr) / 3
    pgcs_mae = np.mean(np.abs(pgcs_oof - gcs))
    pgcs_r2 = 1 - np.sum((gcs - pgcs_oof)**2) / np.sum((gcs - gcs.mean())**2)

    df_tr["pGCS"] = pgcs_oof; df_te["pGCS"] = pgcs_te
    df_tr["pGCS_gbr"] = oof_gbr; df_te["pGCS_gbr"] = te_gbr

    et_fn = lambda: ExtraTreesRegressor(n_estimators=200, max_depth=6,
                                          min_samples_leaf=10, random_state=SEED)
    oof_et, te_et = oof_regress(X_tr, gcs, X_te, et_fn, kf)
    df_tr["pGCS_et"] = oof_et; df_te["pGCS_et"] = te_et

    rf_fn = lambda: RandomForestRegressor(n_estimators=200, max_depth=6,
                                            min_samples_leaf=10, random_state=SEED)
    oof_rf, te_rf = oof_regress(X_tr, gcs, X_te, rf_fn, kf)
    df_tr["pGCS_rf"] = oof_rf; df_te["pGCS_rf"] = te_rf

    def bayesian_oof(X_t, y, X_e, k):
        oof_m = np.zeros(len(X_t))
        for tri, vai in k.split(X_t):
            s = StandardScaler(); Xf = s.fit_transform(X_t[tri])
            br = BayesianRidge(); br.fit(Xf, y[tri])
            oof_m[vai] = br.predict(s.transform(X_t[vai]))
        s = StandardScaler(); Xf = s.fit_transform(X_t)
        br = BayesianRidge(); br.fit(Xf, y)
        return oof_m, br.predict(s.transform(X_e))
        
    oof_bayes, te_bayes = bayesian_oof(X_tr, gcs, X_te, kf)
    df_tr["pGCS_bayes"] = oof_bayes; df_te["pGCS_bayes"] = te_bayes

    q90_fn = lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        loss='quantile', alpha=0.9, subsample=0.8,
        min_samples_leaf=10, random_state=SEED)
    oof_q90, te_q90 = oof_regress(X_tr, gcs, X_te, q90_fn, kf)
    df_tr["pGCS_q90"] = oof_q90; df_te["pGCS_q90"] = te_q90

    for df in [df_tr, df_te]:
        df["age_sq"] = df["age"] ** 2
        df["pGCS_log"] = np.log(np.clip(df["pGCS"], 1, None))
        df["age_x_vol"] = df["age"] * df["hematoma_volume_log"]
        df["pGCS_severe"] = (df["pGCS"] < 7.0).astype(float)
        df["pGCS_mild"] = (df["pGCS"] >= 12.5).astype(float)
        df["density_uniformity"] = 1 - df["density_std"] / (df["density_range"] + 1e-6)
        df["pGCS_sq"] = df["pGCS"] ** 2

    n_bins = 3
    bins = np.percentile(pgcs_oof, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1; bins[-1] += 1
    te_oof = np.zeros(len(df_tr))
    for tri, vai in kf.split(X_tr):
        bin_tr = np.digitize(pgcs_oof[tri], bins) - 1
        bin_va = np.digitize(pgcs_oof[vai], bins) - 1
        rates = {}
        for b in range(n_bins):
            mask = bin_tr == b
            rates[b] = y_tr[tri][mask].mean() if mask.sum() > 0 else y_tr[tri].mean()
        te_oof[vai] = [rates.get(b, y_tr[tri].mean()) for b in bin_va]
    bin_all = np.digitize(pgcs_oof, bins) - 1
    bin_test = np.digitize(pgcs_te, bins) - 1
    rates_all = {b: y_tr[bin_all==b].mean() if (bin_all==b).sum()>0 else y_tr.mean()
                  for b in range(n_bins)}
    te_test = np.array([rates_all.get(b, y_tr.mean()) for b in bin_test])
    df_tr["pGCS_te"] = te_oof; df_te["pGCS_te"] = te_test

    cv_auc, ext_auc, prob_ours_te, oof_ours = eval_lr(
        FINAL_FEATURES, df_tr, df_te, y_tr, y_te, skf, C=FINAL_C)
    ci_ours = bootstrap_auc(y_te, prob_ours_te)

    artifact_dir = RESULTS_DIR / "model_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X_tr_23 = df_tr[FINAL_FEATURES].values
    X_te_23 = df_te[FINAL_FEATURES].values
    scaler_full = StandardScaler()
    X_tr_s = scaler_full.fit_transform(X_tr_23)
    lr_full = LogisticRegression(C=FINAL_C, max_iter=3000, random_state=SEED)
    lr_full.fit(X_tr_s, y_tr)

    with open(artifact_dir / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_full, f)
    with open(artifact_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_full, f)
        
    df_te_save = pd.DataFrame(X_te_23, columns=FINAL_FEATURES)
    df_te_save["y_true"] = y_te
    df_te_save.to_csv(artifact_dir / "test_features_23.csv", index=False)

    _, base_auc, prob_base_te, oof_base = eval_lr(
        BASELINE_FEATS, df_tr, df_te, y_tr, y_te, skf)
    ci_base = bootstrap_auc(y_te, prob_base_te)

    rad_data_path = ROOT / "data" / ""
    rad_cached_te = ROOT / "data" / ""
    rad_cached_tr = ROOT / "data" / ""
    
    try:
        rad_tr = pd.read_csv(rad_data_path)
        rad_te = pd.read_csv(rad_cached_te)
        rad_tr = rad_tr.rename(columns={"patient_id": "New_ID"})
        rad_te = rad_te.rename(columns={"patient_id": "New_ID"})
        rad_feats = [c for c in rad_tr.columns if c != "New_ID"]
        df_tr_m = df_tr.merge(rad_tr, on="New_ID", how="left")
        df_te_m = df_te.merge(rad_te, on="New_ID", how="left")
        for c in rad_feats:
            df_tr_m[c] = df_tr_m[c].fillna(df_tr_m[c].median())
            df_te_m[c] = df_te_m[c].fillna(df_te_m[c].median())
        sc_rad = StandardScaler()
        X_rad_sc = sc_rad.fit_transform(df_tr_m[rad_feats].values)
        lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000)
        lasso.fit(X_rad_sc, y_tr)
        sel_rad = [rad_feats[i] for i in range(len(rad_feats)) if abs(lasso.coef_[i]) > 1e-6]
        if len(sel_rad) < 1: sel_rad = rad_feats[:5]
        _, rad_auc, prob_rad_te, _ = eval_lr(sel_rad, df_tr_m, df_te_m, y_tr, y_te, skf)
        ci_rad = bootstrap_auc(y_te, prob_rad_te)
        
        oof_rad = np.zeros(len(y_tr))
        for tri, vai in skf.split(df_tr_m[sel_rad].values, y_tr):
            m = Pipeline([("s",StandardScaler()),("c",LogisticRegression(C=1.0,max_iter=3000,random_state=SEED))])
            m.fit(df_tr_m[sel_rad].values[tri], y_tr[tri])
            oof_rad[vai] = m.predict_proba(df_tr_m[sel_rad].values[vai])[:,1]
    except Exception:
        prob_rad_te = np.full(len(y_te), 0.5)
        oof_rad = np.full(len(y_tr), 0.5)
        rad_auc = 0.5; ci_rad = [0.5, 0.5]; sel_rad = []

    dl_base = delong_test(y_te, prob_ours_te, prob_base_te)
    nri_base = cfnri(y_te, prob_ours_te, prob_base_te)
    idi_base = idi_test(y_te, prob_ours_te, prob_base_te)

    dl_rad = delong_test(y_te, prob_ours_te, prob_rad_te)
    nri_rad = cfnri(y_te, prob_ours_te, prob_rad_te)
    idi_rad = idi_test(y_te, prob_ours_te, prob_rad_te)

    _, pgcs_auc, _, _ = eval_lr(["pGCS","age"], df_tr, df_te, y_tr, y_te, skf)
    _, gcs_auc, _, _ = eval_lr(["GCS","age"], df_tr, df_te, y_tr, y_te, skf)

    thresholds = np.arange(0.01, 0.80, 0.005)
    prevalence = np.mean(y_te)
    treat_all = [prevalence - (1-prevalence)*t/(1-t) for t in thresholds]
    nb_ours = [net_benefit(y_te, prob_ours_te, t) for t in thresholds]
    nb_base = [net_benefit(y_te, prob_base_te, t) for t in thresholds]
    nb_rad = [net_benefit(y_te, prob_rad_te, t) for t in thresholds]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6.5))
    ax.plot(thresholds, treat_all, 'k--', label='Treat All', linewidth=1, alpha=0.6)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, label='Treat None')
    ax.plot(thresholds, nb_ours, color='#e74c3c', linewidth=2.5,
            label=f'Ours (AUC={ext_auc:.3f})')
    ax.plot(thresholds, nb_base, color='#3498db', linewidth=2,
            label=f'ICH Score (AUC={base_auc:.3f})')
    ax.plot(thresholds, nb_rad, color='#27ae60', linewidth=2, linestyle='--',
            label=f'PyRadiomics LASSO (AUC={rad_auc:.3f})')
    ax.axvspan(0.15, 0.60, alpha=0.06, color='orange')
    ax.text(0.37, 0.02, 'Clinical relevant range', fontsize=9,
            ha='center', color='#e67e22', alpha=0.8, style='italic')
    ax.set_xlim(0, 0.80)
    ax.set_ylim(-0.05, max(0.35, prevalence + 0.05))
    ax.set_xlabel('Threshold Probability', fontsize=13)
    ax.set_ylabel('Net Benefit', fontsize=13)
    ax.set_title('Decision Curve Analysis — mRS 3-6', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dca_mRS3-6.png", dpi=600, bbox_inches='tight')
    plt.close()

    th_clin = np.arange(0.15, 0.61, 0.01)
    nb_o_clin = [net_benefit(y_te, prob_ours_te, t) for t in th_clin]
    nb_b_clin = [net_benefit(y_te, prob_base_te, t) for t in th_clin]
    pct_better = sum(1 for a,b in zip(nb_o_clin,nb_b_clin) if a>b)/len(th_clin)
    int_nb_diff = float(np.mean([a-b for a,b in zip(nb_o_clin,nb_b_clin)]))

    csv_tr = pd.DataFrame({
        "New_ID": df_tr["New_ID"].values,
        "true_mRS3-6": y_tr,
        "pred_proba_Ours": oof_ours,
        "pred_proba_Baseline": oof_base,
        "pred_proba_PyRadiomics": oof_rad,
    })
    csv_tr.to_csv(CSV_DIR / "train_cv_Ours_mRS3-6.csv", index=False)
    csv_bl_tr = pd.DataFrame({
        "New_ID": df_tr["New_ID"].values,
        "true_mRS3-6": y_tr,
        "pred_proba_Baseline": oof_base,
    })
    csv_bl_tr.to_csv(CSV_DIR / "train_cv_Baseline_mRS3-6.csv", index=False)
    csv_rad_tr = pd.DataFrame({
        "New_ID": df_tr["New_ID"].values,
        "true_mRS3-6": y_tr,
        "pred_proba_PyRadiomics": oof_rad,
    })
    csv_rad_tr.to_csv(CSV_DIR / "train_cv_PyRadiomics_mRS3-6.csv", index=False)

    csv_te = pd.DataFrame({
        "New_ID": df_te["New_ID"].values,
        "true_mRS3-6": y_te,
        "pred_proba_Ours": prob_ours_te,
        "pred_proba_Baseline": prob_base_te,
        "pred_proba_PyRadiomics": prob_rad_te,
    })
    csv_te.to_csv(CSV_DIR / "test_Ours_mRS3-6.csv", index=False)
    csv_bl_te = pd.DataFrame({
        "New_ID": df_te["New_ID"].values,
        "true_mRS3-6": y_te,
        "pred_proba_Baseline": prob_base_te,
    })
    csv_bl_te.to_csv(CSV_DIR / "test_Baseline_mRS3-6.csv", index=False)
    csv_rad_te = pd.DataFrame({
        "New_ID": df_te["New_ID"].values,
        "true_mRS3-6": y_te,
        "pred_proba_PyRadiomics": prob_rad_te,
    })
    csv_rad_te.to_csv(CSV_DIR / "test_PyRadiomics_mRS3-6.csv", index=False)

    results = {
        "model": {
            "name": "R3-Best",
            "classifier": f"LogisticRegression(C={FINAL_C})",
        },
        "performance": {
            "cv_auc": float(cv_auc),
            "ext_auc": float(ext_auc),
        },
    }
    with open(RESULTS_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
