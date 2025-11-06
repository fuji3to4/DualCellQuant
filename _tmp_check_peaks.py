import pandas as pd
import numpy as np
from dualcellquant.radial import compute_radial_peak_difference

cp = np.array([60, 70, 80, 90, 100, 110, 120], dtype=float)
vt = np.array([1, 2, 3, 5, 9, 6, 4], dtype=float)
vr = np.array([1, 3, 6, 8, 7, 5, 3], dtype=float)

df = pd.DataFrame({
    'label': [1]*len(cp),
    'center_pct': cp,
    'mean_target': vt,
    'mean_reference': vr
})

for algo in ['first_shoulder','first_local_top','global_max']:
    out = compute_radial_peak_difference(df, min_pct=60, max_pct=120, algo=algo, sg_window=5, sg_poly=2)
    row = out[['max_target_center_pct','max_reference_center_pct','difference_pct']].iloc[0].to_dict()
    print(algo, '->', row)
