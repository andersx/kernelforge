"""Example usage of the high-level model interface."""

import numpy as np
from kernelforge.model import LocalGaussianKRR, GlobalGaussianKRR, load_data

# Generate synthetic data for testing
np.random.seed(42)

# Create simple ethanol-like structures (9 atoms each)
n_structures = 100
natoms = 9

R_list = []
Z_list = []
E_list = []

for i in range(n_structures):
    # Random coordinates
    r = np.random.randn(natoms, 3).astype(np.float64)
    z = np.array([1, 1, 1, 6, 6, 6, 8, 8, 1], dtype=np.int32)  # H3C3O2H

    # Fake energy (distance-based)
    e = float(np.sum(r**2))

    R_list.append(r)
    Z_list.append(z)
    E_list.append(e)

R = R_list
Z = Z_list
E = np.array(E_list)

print("=" * 70)
print("Testing LocalGaussianKRR")
print("=" * 70)

# Test LocalGaussianKRR
model = LocalGaussianKRR(sigma=2.0, regularize=1e-6, elements=[1, 6, 8])
print(f"\nModel: {model}")
print(f"Parameters: {model._get_params()}")

# Fit
print("\nFitting model...")
model.fit(R[:80], Z[:80], E[:80])
print("✓ Model fitted")

# Predict
print("\nPredicting...")
E_pred = model.predict(R[80:], Z[80:])
print(f"✓ Predictions: {E_pred.shape}")
print(f"  True: {E[80:85]}")
print(f"  Pred: {E_pred[:5]}")

# Evaluate
print("\nEvaluating...")
stats = model.evaluate(R, Z, E, n_test=20, seed=42)
print(f"✓ Stats keys: {list(stats.keys())}")

# Save/load
print("\nSaving model...")
model.save("/tmp/test_model.npz")
print("✓ Saved to /tmp/test_model.npz")

print("\nLoading model...")
from kernelforge.model import load_model

model2 = load_model("/tmp/test_model.npz")
print(f"✓ Loaded: {model2}")

# Verify loaded model works (predict on same test set)
test_R = R[85:90]
test_Z = Z[85:90]
E_pred_orig = model.predict(test_R, test_Z)
E_pred2 = model2.predict(test_R, test_Z)
print(f"\nVerifying loaded model...")
print(f"  Original pred: {E_pred_orig}")
print(f"  Loaded pred:   {E_pred2}")
assert np.allclose(E_pred_orig, E_pred2), "Loaded model predictions don't match!"
print("✓ Loaded model verified")

print("\n" + "=" * 70)
print("Testing GlobalGaussianKRR")
print("=" * 70)

# Test GlobalGaussianKRR
model3 = GlobalGaussianKRR(sigma=1.5, regularize=1e-6, descriptor="invdist")
print(f"\nModel: {model3}")

print("\nFitting...")
model3.fit(R[:80], Z[:80], E[:80])
print("✓ Fitted")

print("\nPredicting...")
E_pred3 = model3.predict(R[80:85], Z[80:85])
print(f"✓ Predictions: {E_pred3}")

# Evaluate
stats3 = model3.evaluate(R, Z, E, n_test=20, seed=42)
print(f"✓ Evaluation complete")

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
