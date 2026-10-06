# ============================================================
# CELL 1 — single-stage model comparison, with real XGBoost.
# Run Cell 0 first. Takes a few minutes. Saves table5_results.csv
# and fig3_predictions.csv.
# ============================================================

xgb_grid = {
    "n_estimators": [100, 150],
    "max_depth": [3],
    "learning_rate": [0.1],
    "min_child_weight": [1, 5],   # XGBoost's analogue of min_samples_leaf
}
rf_grid = {"n_estimators": [150], "max_depth": [3, 5, None]}
svr_grid = {"model__C": [1, 10, 100], "model__gamma": ["scale", 0.1]}

def get_model(name):
    if name == "LinearRegression":
        return LinearRegression(), {}, False
    if name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE), rf_grid, False
    if name == "SVR":
        return SVR(kernel="rbf"), svr_grid, True
    if name == "XGBoost":
        return xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0), xgb_grid, False

def nested_cv_eval(estimator, param_grid, X, y, outer_cv, scale=False):
    if scale:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
        grid = param_grid
    else:
        pipe, grid = estimator, param_grid
    search = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1) if grid else pipe
    y_pred = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=-1)
    return r2_score(y, y_pred), np.sqrt(mean_squared_error(y, y_pred)), mean_absolute_error(y, y_pred), y_pred

results_table5 = []
fig3_rows = []  # Sno, Dataset, Model, Actual, Predicted -- best model per dataset only

for mat in materials:
    sub = df[df.Material_name == mat].reset_index(drop=True)
    X = sub[FEATURES].values
    y = sub["Surface roughness"].values
    outer_cv = LeaveOneOut()
    best_r2, best_name, best_pred = -np.inf, None, None
    for model_name in ["LinearRegression", "RandomForest", "SVR", "XGBoost"]:
        model, grid, scale = get_model(model_name)
        r2, rmse, mae, y_pred = nested_cv_eval(model, grid, X, y, outer_cv, scale=scale)
        results_table5.append({"Dataset": mat, "Model": model_name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"{mat:10s} {model_name:16s} R2={r2:7.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
        if r2 > best_r2:
            best_r2, best_name, best_pred = r2, model_name, y_pred
    for sno, actual, pred in zip(sub["Sno"], y, best_pred):
        fig3_rows.append({"Sno": sno, "Dataset": mat, "Model": best_name, "Actual": actual, "Predicted": pred})

# Unified dataset
ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])
X_unified = np.hstack([df[FEATURES].values, mat_dum])
y_unified = df["Surface roughness"].values
outer_cv_u = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

best_r2, best_name, best_pred = -np.inf, None, None
for model_name in ["LinearRegression", "RandomForest", "SVR", "XGBoost"]:
    model, grid, scale = get_model(model_name)
    r2, rmse, mae, y_pred = nested_cv_eval(model, grid, X_unified, y_unified, outer_cv_u, scale=scale)
    results_table5.append({"Dataset": "Unified", "Model": model_name, "R2": r2, "RMSE": rmse, "MAE": mae})
    print(f"{'Unified':10s} {model_name:16s} R2={r2:7.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    if r2 > best_r2:
        best_r2, best_name, best_pred = r2, model_name, y_pred
for sno, actual, pred in zip(df["Sno"], y_unified, best_pred):
    fig3_rows.append({"Sno": sno, "Dataset": "Unified", "Model": best_name, "Actual": actual, "Predicted": pred})

table5_df = pd.DataFrame(results_table5)
table5_df.to_csv("table5_results.csv", index=False)
pd.DataFrame(fig3_rows).to_csv("fig3_predictions.csv", index=False)

print("\nSaved table5_results.csv and fig3_predictions.csv")
print(table5_df.round(3).to_string(index=False))
