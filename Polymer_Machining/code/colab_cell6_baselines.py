# ============================================================
# CELL 6
#   (a) a quadratic Response Surface Methodology (RSM) model,
#       matching the classical approach
#   (b) the classical geometric turning-roughness formula,
#       Ra = f^2 / (32 * r), using the real tool nose radius
#       (0.8 mm, CNMG 120408 insert)
# Run Cell 0 first. Saves table_baselines.csv.
# ============================================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as OLS

NOSE_RADIUS_MM = 0.8

def geometric_baseline(feed_mm_per_rev):
    # Ra in micrometres; feed and radius both in mm
    return (feed_mm_per_rev ** 2 / (32 * NOSE_RADIUS_MM)) * 1000

def rsm_quadratic_nested_cv(X, y, outer_cv):
    y_pred_all = np.zeros(len(y))
    poly = PolynomialFeatures(degree=2, include_bias=False)
    for train_idx, test_idx in outer_cv.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]
        Xtr_poly = poly.fit_transform(Xtr)
        Xte_poly = poly.transform(Xte)
        model = OLS().fit(Xtr_poly, ytr)
        y_pred_all[test_idx] = model.predict(Xte_poly)
    return y_pred_all

baseline_rows = []

for mat in materials:
    sub = df[df.Material_name == mat].reset_index(drop=True)
    X = sub[FEATURES].values
    y = sub["Surface roughness"].values

    # (a) geometric baseline -- no fitting, pure physics, evaluated directly
    geo_pred = geometric_baseline(sub["Feed"].values)
    geo_r2 = r2_score(y, geo_pred)
    geo_rmse = np.sqrt(mean_squared_error(y, geo_pred))
    baseline_rows.append({"Dataset": mat, "Baseline": "Geometric (Ra=f^2/32r)", "R2": geo_r2, "RMSE": geo_rmse})
    print(f"{mat:10s} geometric      R2={geo_r2:8.3f}  RMSE={geo_rmse:.3f}")

    # (b) RSM quadratic, nested CV (LOOCV, matching Table 4/5 protocol)
    rsm_pred = rsm_quadratic_nested_cv(X, y, LeaveOneOut())
    rsm_r2 = r2_score(y, rsm_pred)
    rsm_rmse = np.sqrt(mean_squared_error(y, rsm_pred))
    baseline_rows.append({"Dataset": mat, "Baseline": "RSM quadratic", "R2": rsm_r2, "RMSE": rsm_rmse})
    print(f"{mat:10s} RSM quadratic  R2={rsm_r2:8.3f}  RMSE={rsm_rmse:.3f}")

# Unified: geometric baseline still just uses feed (material-agnostic, deliberately -- it
# has no way to represent material identity, which is itself informative to report);
# RSM quadratic gets material one-hot flags added as linear terms only (no interactions
# with them, to keep it a recognisable RSM model rather than a black box).
ohe = OneHotEncoder(sparse_output=False)
mat_dum = ohe.fit_transform(df[["Material_name"]])

geo_pred_u = geometric_baseline(df["Feed"].values)
geo_r2_u = r2_score(df["Surface roughness"].values, geo_pred_u)
geo_rmse_u = np.sqrt(mean_squared_error(df["Surface roughness"].values, geo_pred_u))
baseline_rows.append({"Dataset": "Unified", "Baseline": "Geometric (Ra=f^2/32r)", "R2": geo_r2_u, "RMSE": geo_rmse_u})
print(f"{'Unified':10s} geometric      R2={geo_r2_u:8.3f}  RMSE={geo_rmse_u:.3f}")

X_u_params = df[FEATURES].values
poly_u = PolynomialFeatures(degree=2, include_bias=False)
X_u_poly = poly_u.fit_transform(X_u_params)
X_u_full = np.hstack([X_u_poly, mat_dum])
y_u = df["Surface roughness"].values
outer_cv_u = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rsm_pred_u = np.zeros(len(y_u))
for train_idx, test_idx in outer_cv_u.split(X_u_full):
    model = OLS().fit(X_u_full[train_idx], y_u[train_idx])
    rsm_pred_u[test_idx] = model.predict(X_u_full[test_idx])
rsm_r2_u = r2_score(y_u, rsm_pred_u)
rsm_rmse_u = np.sqrt(mean_squared_error(y_u, rsm_pred_u))
baseline_rows.append({"Dataset": "Unified", "Baseline": "RSM quadratic", "R2": rsm_r2_u, "RMSE": rsm_rmse_u})
print(f"{'Unified':10s} RSM quadratic  R2={rsm_r2_u:8.3f}  RMSE={rsm_rmse_u:.3f}")

baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv("table_baselines.csv", index=False)
print("\nSaved table_baselines.csv")
print(baseline_df.round(3).to_string(index=False))
