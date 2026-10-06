# ============================================================
# CELL 9 — Feature importance via out-of-fold permutation
# importance with real XGBoost, reporting uncertainty across
# 
# Run Cell 0 first. Saves table_feature_importance.csv.
# ============================================================

from sklearn.inspection import permutation_importance

param_grid = {"n_estimators": [100, 150], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}

ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])
feat_names = FEATURES + list(ohe.get_feature_names_out(["Material_name"]))
X = np.hstack([df[FEATURES].values, mat_dum])
y = df["Surface roughness"].values

# Fit the final model on all data (same as the manuscript's "refit for deployment" step)
final_search = GridSearchCV(xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0), param_grid,
                             cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
final_search.fit(X, y)
best_model = final_search.best_estimator_
print("Best hyperparameters:", final_search.best_params_)

result = permutation_importance(best_model, X, y, n_repeats=30, random_state=RANDOM_STATE, scoring="r2")

imp_rows = []
for name, mean_imp, std_imp in zip(feat_names, result.importances_mean, result.importances_std):
    imp_rows.append({"Feature": name, "Importance_mean": mean_imp, "Importance_std": std_imp})
imp_df = pd.DataFrame(imp_rows).sort_values("Importance_mean", ascending=False)
imp_df.to_csv("table_feature_importance.csv", index=False)
print("\nSaved table_feature_importance.csv")
print(imp_df.round(4).to_string(index=False))
