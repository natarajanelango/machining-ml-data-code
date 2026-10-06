# ============================================================
# CELL 4 — original leaky two-stage model,
# XGBoost classifier + regressor, to fully
# match the architecture described in Natarajan et al. (2025).
# Run Cell 0 first. Saves table3_results.csv.
# ============================================================

def run_leaky_pipeline_xgb(X_params, y):
    bin_edges = np.quantile(y, [0, 0.5, 1.0])
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    y_class = np.digitize(y, bin_edges[1:-1])

    X_clf = np.hstack([X_params, y.reshape(-1, 1)])
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
                             verbosity=0, eval_metric="logloss")
    clf.fit(X_clf, y_class)
    class_pred = clf.predict(X_clf)

    X_reg = np.hstack([X_params, class_pred.reshape(-1, 1), y.reshape(-1, 1)])
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, verbosity=0)
    reg.fit(X_reg, y)
    y_pred = reg.predict(X_reg)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return r2, rmse

results_table3 = []
for mat in materials:
    sub = df[df.Material_name == mat]
    X = sub[FEATURES].values
    y = sub["Surface roughness"].values
    r2, rmse = run_leaky_pipeline_xgb(X, y)
    results_table3.append({"Dataset": mat, "R2": r2, "RMSE": rmse})
    print(f"{mat:10s}: R2={r2:.3f}  RMSE={rmse:.3f}")

ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])
X_unified = np.hstack([df[FEATURES].values, mat_dum])
y_unified = df["Surface roughness"].values
r2u, rmseu = run_leaky_pipeline_xgb(X_unified, y_unified)
results_table3.append({"Dataset": "Unified", "R2": r2u, "RMSE": rmseu})
print(f"{'Unified':10s}: R2={r2u:.3f}  RMSE={rmseu:.3f}")

table3_df = pd.DataFrame(results_table3)
table3_df.to_csv("table3_results.csv", index=False)
print("\nSaved table3_results.csv")
print(table3_df.round(3).to_string(index=False))
