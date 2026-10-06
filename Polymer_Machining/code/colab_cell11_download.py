# ============================================================
# CELL 11 — Run this last. Bundles every results CSV that exists
# into one zip file and downloads it. Just give me that one file.
# ============================================================

import os, zipfile
from google.colab import files

possible_files = [
    "table3_results.csv",
    "table4_results.csv",
    "table4_classifier_raw.csv",
    "table4_classifier_performance.csv",
    "table5_results.csv",
    "table6_results.csv",
    "fig3_predictions.csv",
    "table_baselines.csv",
    "table_uncertainty.csv",
    "table_uncertainty_raw_repeats.csv",
    "table_ablation_material_id.csv",
    "table_ablation_cross_material_transfer.csv",
    "table_feature_importance.csv",
    "table_bimodality_bandwidth.csv",
    "table_bimodality_bic.csv",
]
found = [f for f in possible_files if os.path.exists(f)]
missing = [f for f in possible_files if f not in found]

print("Found:", found)
if missing:
    print("Not found (skipped, fine if you didn't run that cell):", missing)

zip_name = "xgboost_reanalysis_results.zip"
with zipfile.ZipFile(zip_name, "w") as zf:
    for f in found:
        zf.write(f)

print(f"\nSaved {zip_name} with {len(found)} file(s). Downloading now...")
files.download(zip_name)
