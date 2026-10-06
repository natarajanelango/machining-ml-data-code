# ============================================================
# CELL 3 — unified model vs. four separate
# material-specific models, matched folds, real XGBoost.
# Run Cell 0 first. Saves table6_results.csv.
# ============================================================

param_grid = {"n_estimators": [100, 150], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}

ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])
X_unified = np.hstack([df[FEATURES].values, mat_dum])
y = df["Surface roughness"].values

outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
splits = list(outer_cv.split(df))

unified_pred = np.zeros(len(df))
separate_pred = np.zeros(len(df))

for fold_i, (train_idx, test_idx) in enumerate(splits):
    search_u = GridSearchCV(xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
                             param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    search_u.fit(X_unified[train_idx], y[train_idx])
    unified_pred[test_idx] = search_u.predict(X_unified[test_idx])

    for mat in materials:
        mat_train_idx = [i for i in train_idx if df.loc[i, "Material_name"] == mat]
        mat_test_idx = [i for i in test_idx if df.loc[i, "Material_name"] == mat]
        if not mat_test_idx:
            continue
        Xtr = df.loc[mat_train_idx, FEATURES].values
        ytr = df.loc[mat_train_idx, "Surface roughness"].values
        Xte = df.loc[mat_test_idx, FEATURES].values
        search_s = GridSearchCV(xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
                                 param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
        search_s.fit(Xtr, ytr)
        separate_pred[mat_test_idx] = search_s.predict(Xte)
    print(f"fold {fold_i} done")

rows = []
r2u, rmseu = r2_score(y, unified_pred), np.sqrt(mean_squared_error(y, unified_pred))
r2s, rmses = r2_score(y, separate_pred), np.sqrt(mean_squared_error(y, separate_pred))
rows.append({"Dataset": "Unified (pooled)", "Unified_R2": r2u, "Unified_RMSE": rmseu,
             "Separate_R2": r2s, "Separate_RMSE": rmses})
for mat in materials:
    idx = (df.Material_name == mat).values
    ru2 = r2_score(y[idx], unified_pred[idx]); ru_rmse = np.sqrt(mean_squared_error(y[idx], unified_pred[idx]))
    rs2 = r2_score(y[idx], separate_pred[idx]); rs_rmse = np.sqrt(mean_squared_error(y[idx], separate_pred[idx]))
    rows.append({"Dataset": mat, "Unified_R2": ru2, "Unified_RMSE": ru_rmse,
                 "Separate_R2": rs2, "Separate_RMSE": rs_rmse})

table6_df = pd.DataFrame(rows)
table6_df.to_csv("table6_results.csv", index=False)
print("\nSaved table6_results.csv")
print(table6_df.round(3).to_string(index=False))
