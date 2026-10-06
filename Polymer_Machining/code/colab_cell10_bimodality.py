# ============================================================
# CELL 10 — Is Delrin's Ra distribution genuinely bimodal, or an
# artefact of the three discrete factor levels and KDE bandwidth
# choice? Two checks:
#  (a) KDE bandwidth sensitivity 
#  (b) A formal model comparison: 
# Run Cell 0 first. Saves table_bimodality.csv.
# ============================================================

from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

delrin_ra = df[df.Material_name == "Delrin"]["Surface roughness"].values.reshape(-1, 1)

# (a) bandwidth sensitivity
bw_rows = []
x_grid = np.linspace(delrin_ra.min() - 0.5, delrin_ra.max() + 0.5, 500)
for bw in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, "scott", "silverman"]:
    kde = gaussian_kde(delrin_ra.ravel(), bw_method=bw if isinstance(bw, float) else bw)
    density = kde(x_grid)
    # count local maxima (peaks) in the density curve
    peaks = ((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])).sum()
    bw_rows.append({"bandwidth": bw, "n_peaks_detected": int(peaks)})
    print(f"bandwidth={bw!s:10s}  peaks detected={peaks}")

# (b) 1-component vs 2-component Gaussian mixture, BIC comparison
gmm1 = GaussianMixture(n_components=1, random_state=RANDOM_STATE).fit(delrin_ra)
gmm2 = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(delrin_ra)
bic1, bic2 = gmm1.bic(delrin_ra), gmm2.bic(delrin_ra)
print(f"\n1-component BIC: {bic1:.2f}")
print(f"2-component BIC: {bic2:.2f}")
print(f"Lower BIC wins. Difference (BIC1 - BIC2) = {bic1 - bic2:+.2f}  "
      f"({'favours 2 components (bimodal)' if bic1 > bic2 else 'favours 1 component (unimodal)'})")

bw_df = pd.DataFrame(bw_rows)
bic_df = pd.DataFrame([{"n_components": 1, "BIC": bic1}, {"n_components": 2, "BIC": bic2}])
bw_df.to_csv("table_bimodality_bandwidth.csv", index=False)
bic_df.to_csv("table_bimodality_bic.csv", index=False)
print("\nSaved table_bimodality_bandwidth.csv and table_bimodality_bic.csv")
