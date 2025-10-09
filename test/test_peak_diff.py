"""
Test the compute_radial_peak_difference function with pixel and um conversion
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'e:\\Data\\App\\Python\\dualCellQuant')

from dualCellQuant import compute_radial_peak_difference

# Create sample radial profile data
radial_data = {
    'label': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    'center_pct': [50.0, 70.0, 90.0, 110.0, 130.0, 50.0, 70.0, 90.0, 110.0, 130.0],
    'mean_target': [10.0, 15.0, 25.0, 20.0, 18.0, 8.0, 12.0, 18.0, 16.0, 14.0],
    'mean_reference': [5.0, 8.0, 12.0, 15.0, 10.0, 6.0, 9.0, 14.0, 11.0, 8.0],
}

radial_df = pd.DataFrame(radial_data)

# Create sample quantification data with area information
# area_and_px = π * r^2, so if r = 10px, area = 314.16px^2
# If pixel size is 0.5μm, then r = 5μm, area = 78.54μm^2
quant_data = {
    'label': [1, 2],
    'area_and_px': [314.16, 78.54],  # r_eq = 10px and 5px
    'area_and_um2': [78.54, 19.635], # r_eq = 5μm and 2.5μm
}

quant_df = pd.DataFrame(quant_data)

print("Sample radial profile data:")
print(radial_df)
print("\n" + "="*80 + "\n")

print("Sample quantification data:")
print(quant_df)
print("\n" + "="*80 + "\n")

# Test without quantification data (should only show % values)
print("Test 1: Without quantification data (only % values)")
result1 = compute_radial_peak_difference(radial_df, None, min_pct=60.0, max_pct=120.0)
print(result1)
print("\n" + "="*80 + "\n")

# Test with quantification data (should show %, px, and um values)
print("Test 2: With quantification data (%, px, and μm values)")
result2 = compute_radial_peak_difference(radial_df, quant_df, min_pct=60.0, max_pct=120.0)
print(result2)
print("\n" + "="*80 + "\n")

print("Validation:")
for idx, row in result2.iterrows():
    label = row['label']
    # Percentage
    max_t_pct = row['max_target_center_pct']
    max_r_pct = row['max_reference_center_pct']
    diff_pct = row['difference_pct']
    # Pixels
    max_t_px = row['max_target_px']
    max_r_px = row['max_reference_px']
    diff_px = row['difference_px']
    # Micrometers
    max_t_um = row['max_target_um']
    max_r_um = row['max_reference_um']
    diff_um = row['difference_um']
    
    print(f"\nLabel {label}:")
    print(f"  % : Target peaks at {max_t_pct:.1f}%, Reference peaks at {max_r_pct:.1f}%, Difference = {diff_pct:.1f}%")
    print(f"  px: Target peaks at {max_t_px:.2f}px, Reference peaks at {max_r_px:.2f}px, Difference = {diff_px:.2f}px")
    print(f"  μm: Target peaks at {max_t_um:.2f}μm, Reference peaks at {max_r_um:.2f}μm, Difference = {diff_um:.2f}μm")

print("\n✓ Test completed successfully!")

# Expected for Label 1 (r_eq = 10px = 5μm):
#   Target at 90% = 9.0px = 4.5μm
#   Reference at 110% = 11.0px = 5.5μm
#   Difference = -20% = -2.0px = -1.0μm

# Expected for Label 2 (r_eq = 5px = 2.5μm):
#   Target at 90% = 4.5px = 2.25μm
#   Reference at 90% = 4.5px = 2.25μm
#   Difference = 0% = 0.0px = 0.0μm

