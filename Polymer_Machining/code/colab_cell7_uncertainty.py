# ============================================================
# CELL 7 — Uncertainty on model performance
# Repeats a shuffled outer cross-validation with 20 different
# random seeds and reports mean, std, and 95% interval for: the
# winning single-stage model, the corrected two-stage model, and
# their paired difference on identical held-out folds each repeat.
#
# Note: per-material results use repeated 3-fold CV (9 samples per
# fold) rather than the leave-one-out 
# Run Cell 0 first. 
# Saves table_uncertainty.csv and table_uncertainty_raw_repeats.csv.
# ============================================================

N_REPEATS = 20

xgb_reg_grid = {"n_estimators": [100, 150], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}
xgb_clf_grid = {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}

def single_stage_r2(X, y, outer_cv):
    search = GridSearchCV(xgb.XGBRegressor(random_state=0, verbosity=0), xgb_reg_grid,
                           cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    pred = cross_val_predict(search, X, y, cv=outer_cv, n_jobs=-1)
    return r2_score(y, pred)

def two_stage_r2(X, y, outer_cv):
    n = len(y)
    pred_all = np.zeros(n)
    for train_idx, test_idx in outer_cv.split(X):
        y_train = y[train_idx]
        edges = np.quantile(y_train, [0, 0.5, 1.0]); edges[0], edges[-1] = -np.inf, np.inf
        y_class_train = np.digitize(y_train, edges[1:-1])
        Xtr, Xte = X[train_idx], X[test_idx]
        inner_cv = KFold(n_splits=min(3, len(train_idx) - 1), shuffle=True, random_state=0)
        clf_s = GridSearchCV(xgb.XGBClassifier(random_state=0, verbosity=0, eval_metric="logloss"),
                              xgb_clf_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        oof_class = cross_val_predict(clf_s, Xtr, y_class_train, cv=inner_cv, n_jobs=-1)
        clf_f = GridSearchCV(xgb.XGBClassifier(random_state=0, verbosity=0, eval_metric="logloss"),
                              xgb_clf_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        clf_f.fit(Xtr, y_class_train)
        class_test = clf_f.predict(Xte)
        reg_s = GridSearchCV(xgb.XGBRegressor(random_state=0, verbosity=0), xgb_reg_grid,
                              cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1)
        reg_s.fit(np.hstack([Xtr, oof_class.reshape(-1, 1)]), y_train)
        pred_all[test_idx] = reg_s.predict(np.hstack([Xte, class_test.reshape(-1, 1)]))
    return r2_score(y, pred_all)

uncertainty_rows = []
raw_repeats = []

for mat in materials + ["Unified"]:
    if mat == "Unified":
        ohe = OneHotEncoder(sparse_output=False)
        mat_dum = ohe.fit_transform(df[["Material_name"]])
        X = np.hstack([df[FEATURES].values, mat_dum])
        y = df["Surface roughness"].values
        n_splits = 5
    else:
        sub = df[df.Material_name == mat].reset_index(drop=True)
        X = sub[FEATURES].values
        y = sub["Surface roughness"].values
        n_splits = 3  # 27 samples / 3 folds = 9 per fold

    ss_scores, ts_scores, diffs = [], [], []
    for seed in range(N_REPEATS):
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        ss_r2 = single_stage_r2(X, y, outer_cv)
        ts_r2 = two_stage_r2(X, y, outer_cv)
        ss_scores.append(ss_r2); ts_scores.append(ts_r2); diffs.append(ss_r2 - ts_r2)
        raw_repeats.append({"Dataset": mat, "seed": seed, "single_stage_R2": ss_r2, "two_stage_R2": ts_r2})

    ss_scores, ts_scores, diffs = map(np.array, (ss_scores, ts_scores, diffs))
    uncertainty_rows.append({
        "Dataset": mat,
        "single_stage_R2_mean": ss_scores.mean(), "single_stage_R2_std": ss_scores.std(),
        "two_stage_R2_mean": ts_scores.mean(), "two_stage_R2_std": ts_scores.std(),
        "diff_mean": diffs.mean(), "diff_std": diffs.std(),
        "diff_95CI_lower": np.percentile(diffs, 2.5), "diff_95CI_upper": np.percentile(diffs, 97.5),
    })
    print(f"{mat:10s} single-stage R2={ss_scores.mean():.3f}+/-{ss_scores.std():.3f}  "
          f"two-stage R2={ts_scores.mean():.3f}+/-{ts_scores.std():.3f}  "
          f"diff={diffs.mean():+.3f} [{np.percentile(diffs,2.5):+.3f}, {np.percentile(diffs,97.5):+.3f}]")

uncertainty_df = pd.DataFrame(uncertainty_rows)
uncertainty_df.to_csv("table_uncertainty.csv", index=False)
pd.DataFrame(raw_repeats).to_csv("table_uncertainty_raw_repeats.csv", index=False)
print("\nSaved table_uncertainty.csv and table_uncertainty_raw_repeats.csv")
print(uncertainty_df.round(3).to_string(index=False))
