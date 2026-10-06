# ============================================================
# CELL 8 — Is the unified model's benefit genuine cross-material
# transfer
#
# Two things are tested:
#  (a) Unified model WITH material identity vs WITHOUT it.
#  (b) For each material, a model trained on the OTHER THREE
#      materials only 
# Run Cell 0 first. Saves table_ablation.csv.
# ============================================================

param_grid = {"n_estimators": [100, 150], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}

ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])
X_with_id = np.hstack([df[FEATURES].values, mat_dum])
X_no_id = df[FEATURES].values
y = df["Surface roughness"].values

outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def nested_r2(X, y, outer_cv):
    search = GridSearchCV(xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0), param_grid,
                           cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    pred = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=-1)
    return r2_score(y, pred), pred

r2_with_id, pred_with_id = nested_r2(X_with_id, y, outer_cv)
r2_no_id, pred_no_id = nested_r2(X_no_id, y, outer_cv)
print(f"Unified WITH material identity:    R2={r2_with_id:.3f}")
print(f"Unified WITHOUT material identity: R2={r2_no_id:.3f}")

ablation_rows = [{
    "Comparison": "Unified pooled",
    "With_material_ID_R2": r2_with_id,
    "Without_material_ID_R2": r2_no_id,
}]
for mat in materials:
    idx = (df.Material_name == mat).values
    ablation_rows.append({
        "Comparison": f"Unified pooled -- {mat} only",
        "With_material_ID_R2": r2_score(y[idx], pred_with_id[idx]),
        "Without_material_ID_R2": r2_score(y[idx], pred_no_id[idx]),
    })

# (b) leave-one-material-out transfer test: train on the other three, no material ID
print("\nCross-material transfer (trained on other 3 materials only, no material ID):")
transfer_rows = []
for held_out in materials:
    train_df = df[df.Material_name != held_out]
    test_df = df[df.Material_name == held_out]
    Xtr = train_df[FEATURES].values
    ytr = train_df["Surface roughness"].values
    Xte = test_df[FEATURES].values
    yte = test_df["Surface roughness"].values
    search = GridSearchCV(xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0), param_grid,
                           cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    search.fit(Xtr, ytr)
    pred = search.predict(Xte)
    r2 = r2_score(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    transfer_rows.append({"Held_out_material": held_out, "Transfer_R2": r2, "Transfer_RMSE": rmse})
    print(f"  Held out {held_out:12s}: R2={r2:7.3f}  RMSE={rmse:.3f}")

ablation_df = pd.DataFrame(ablation_rows)
transfer_df = pd.DataFrame(transfer_rows)
ablation_df.to_csv("table_ablation_material_id.csv", index=False)
transfer_df.to_csv("table_ablation_cross_material_transfer.csv", index=False)
print("\nSaved table_ablation_material_id.csv and table_ablation_cross_material_transfer.csv")
print(ablation_df.round(3).to_string(index=False))
