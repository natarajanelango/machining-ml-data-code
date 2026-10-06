# ============================================================
# CELL 2 — Table 4: corrected two-stage model with XGBoost,
# now also reporting classifier performance.
# Run Cell 0 first. Saves table4_results.csv and
# table4_classifier_performance.csv.
# ============================================================

clf_grid = {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}
reg_grid = {"n_estimators": [100, 150], "max_depth": [3], "learning_rate": [0.1], "min_child_weight": [1, 5]}

from sklearn.metrics import accuracy_score, confusion_matrix

def corrected_two_stage_xgb(X, y, outer_cv, inner_splits=3):
    n = len(y)
    y_pred_all = np.zeros(n)
    clf_records = []  # per outer fold: accuracy, confusion matrix, class balance
    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        y_train = y[train_idx]
        bin_edges = np.quantile(y_train, [0, 0.5, 1.0])
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        y_class_train = np.digitize(y_train, bin_edges[1:-1])

        y_test = y[test_idx]
        y_class_test = np.digitize(y_test, bin_edges[1:-1])  # true class of held-out point(s),
                                                               # using the training-fold thresholds

        X_train, X_test = X[train_idx], X[test_idx]
        inner_cv = KFold(n_splits=min(inner_splits, len(train_idx) - 1),
                          shuffle=True, random_state=RANDOM_STATE)

        clf_search = GridSearchCV(
            xgb.XGBClassifier(random_state=RANDOM_STATE, verbosity=0, eval_metric="logloss"),
            clf_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        oof_class_train = cross_val_predict(clf_search, X_train, y_class_train, cv=inner_cv, n_jobs=-1)

        clf_final = GridSearchCV(
            xgb.XGBClassifier(random_state=RANDOM_STATE, verbosity=0, eval_metric="logloss"),
            clf_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)
        clf_final.fit(X_train, y_class_train)
        class_pred_test = clf_final.predict(X_test)

        # record classifier performance on this fold's held-out point(s)
        for true_c, pred_c in zip(y_class_test, class_pred_test):
            clf_records.append({"fold": fold_i, "true_class": true_c, "pred_class": pred_c,
                                 "correct": int(true_c == pred_c)})

        X_reg_train = np.hstack([X_train, oof_class_train.reshape(-1, 1)])
        X_reg_test = np.hstack([X_test, class_pred_test.reshape(-1, 1)])
        reg_search = GridSearchCV(
            xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            reg_grid, cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1)
        reg_search.fit(X_reg_train, y_train)
        y_pred_all[test_idx] = reg_search.predict(X_reg_test)
    return y_pred_all, clf_records

results_table4 = []
all_clf_records = []
for mat in materials:
    sub = df[df.Material_name == mat].reset_index(drop=True)
    X = sub[FEATURES].values
    y = sub["Surface roughness"].values
    outer_cv = LeaveOneOut()
    pred, clf_records = corrected_two_stage_xgb(X, y, outer_cv)
    for r in clf_records:
        r["Dataset"] = mat
    all_clf_records.extend(clf_records)
    r2 = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    results_table4.append({"Dataset": mat, "R2": r2, "RMSE": rmse, "MAE": mae})
    print(f"{mat:10s}: R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])
X_unified = np.hstack([df[FEATURES].values, mat_dum])
y_unified = df["Surface roughness"].values
outer_cv_u = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
pred_u, clf_records_u = corrected_two_stage_xgb(X_unified, y_unified, outer_cv_u)
for r in clf_records_u:
    r["Dataset"] = "Unified"
all_clf_records.extend(clf_records_u)
r2u = r2_score(y_unified, pred_u)
rmseu = np.sqrt(mean_squared_error(y_unified, pred_u))
maeu = mean_absolute_error(y_unified, pred_u)
results_table4.append({"Dataset": "Unified", "R2": r2u, "RMSE": rmseu, "MAE": maeu})
print(f"{'Unified':10s}: R2={r2u:.3f}  RMSE={rmseu:.3f}  MAE={maeu:.3f}")

table4_df = pd.DataFrame(results_table4)
table4_df.to_csv("table4_results.csv", index=False)

clf_df = pd.DataFrame(all_clf_records)
clf_df.to_csv("table4_classifier_raw.csv", index=False)

# summary: accuracy and confusion matrix per dataset
clf_summary_rows = []
for mat in materials + ["Unified"]:
    sub = clf_df[clf_df.Dataset == mat]
    acc = sub["correct"].mean()
    cm = confusion_matrix(sub["true_class"], sub["pred_class"], labels=[0, 1])
    class_balance = sub["true_class"].value_counts().to_dict()
    clf_summary_rows.append({
        "Dataset": mat, "Classifier_accuracy": acc,
        "n_class0": class_balance.get(0, 0), "n_class1": class_balance.get(1, 0),
        "confusion_matrix_[[TN,FP],[FN,TP]]": cm.tolist(),
    })
clf_summary_df = pd.DataFrame(clf_summary_rows)
clf_summary_df.to_csv("table4_classifier_performance.csv", index=False)

print("\nSaved table4_results.csv, table4_classifier_raw.csv, table4_classifier_performance.csv")
print(table4_df.round(3).to_string(index=False))
print()
print(clf_summary_df.to_string(index=False))
