"""Fault detection using SPC charts and residual monitoring.

Demonstrates:
1. Using SPCChart for statistical process control
2. Setting a baseline from normal operation data
3. Injecting a fault (step change) and detecting it
4. Showing EWMA and CUSUM alarm triggering

Run: python examples/12_fault_detection.py
"""

from __future__ import annotations

import numpy as np

from reactor_twin.digital_twin import SPCChart

np.random.seed(42)


def main() -> None:
    """Run fault detection example."""
    print("=" * 60)
    print("Example 12: Fault Detection with SPC Charts")
    print("=" * 60)

    # 1. Generate normal operation data
    print("\n1. Generating normal operation data (200 samples)...")
    num_vars = 2  # C_A, C_B
    num_baseline = 200
    num_test = 100

    # Normal operation: C_A ~ 0.5, C_B ~ 0.5 (steady state with noise)
    normal_mean = np.array([0.5, 0.5])
    normal_std = np.array([0.02, 0.02])

    baseline_data = normal_mean + normal_std * np.random.randn(num_baseline, num_vars)
    print(f"   Baseline data shape: {baseline_data.shape}")
    print(f"   C_A mean: {baseline_data[:, 0].mean():.4f} +/- {baseline_data[:, 0].std():.4f}")
    print(f"   C_B mean: {baseline_data[:, 1].mean():.4f} +/- {baseline_data[:, 1].std():.4f}")

    # 2. Set up SPC chart
    print("\n2. Setting up SPC chart (EWMA + CUSUM)...")
    spc = SPCChart(
        num_vars=num_vars,
        ewma_lambda=0.2,
        ewma_L=3.0,
        cusum_k=0.5,
        cusum_h=5.0,
    )
    spc.set_baseline(baseline_data)
    print(f"   Baseline mean: {spc.mean}")
    print(f"   Baseline std:  {spc.std}")
    print(f"   EWMA lambda: {spc.ewma_lambda}, L: {spc.ewma_L}")
    print(f"   CUSUM k: {spc.cusum_k}, h: {spc.cusum_h}")

    # 3. Generate test data: normal for first 50 samples, fault after
    print("\n3. Generating test data with fault injection at sample 50...")
    fault_onset = 50
    fault_magnitude = 0.1  # Step change in C_A

    test_data = normal_mean + normal_std * np.random.randn(num_test, num_vars)
    # Inject fault: step change in C_A after fault_onset
    test_data[fault_onset:, 0] += fault_magnitude

    print(f"   Normal phase: samples 0-{fault_onset - 1}")
    print(f"   Fault phase:  samples {fault_onset}-{num_test - 1}")
    print(f"   Fault: C_A shifts by +{fault_magnitude} mol/L")

    # 4. Run SPC monitoring
    print("\n4. Running SPC monitoring...")
    ewma_alarms = []
    cusum_alarms = []
    first_ewma_alarm = None
    first_cusum_alarm = None

    for i in range(num_test):
        result = spc.update(test_data[i])

        ewma_any = result["ewma_alarm"].any()
        cusum_any = result["cusum_alarm"].any()

        ewma_alarms.append(ewma_any)
        cusum_alarms.append(cusum_any)

        if ewma_any and first_ewma_alarm is None:
            first_ewma_alarm = i
        if cusum_any and first_cusum_alarm is None:
            first_cusum_alarm = i

    # 5. Results
    print("\n5. Detection results:")
    print(f"   Fault onset:             sample {fault_onset}")
    if first_ewma_alarm is not None:
        print(f"   First EWMA alarm:        sample {first_ewma_alarm} (delay = {first_ewma_alarm - fault_onset} samples)")
    else:
        print(f"   First EWMA alarm:        not triggered")
    if first_cusum_alarm is not None:
        print(f"   First CUSUM alarm:       sample {first_cusum_alarm} (delay = {first_cusum_alarm - fault_onset} samples)")
    else:
        print(f"   First CUSUM alarm:       not triggered")

    # Count alarms
    num_ewma_normal = sum(ewma_alarms[:fault_onset])
    num_cusum_normal = sum(cusum_alarms[:fault_onset])
    num_ewma_fault = sum(ewma_alarms[fault_onset:])
    num_cusum_fault = sum(cusum_alarms[fault_onset:])

    print(f"\n   False alarms (before fault):")
    print(f"     EWMA:  {num_ewma_normal}/{fault_onset} = {100 * num_ewma_normal / fault_onset:.1f}%")
    print(f"     CUSUM: {num_cusum_normal}/{fault_onset} = {100 * num_cusum_normal / fault_onset:.1f}%")
    print(f"   True alarms (after fault):")
    print(f"     EWMA:  {num_ewma_fault}/{num_test - fault_onset} = {100 * num_ewma_fault / (num_test - fault_onset):.1f}%")
    print(f"     CUSUM: {num_cusum_fault}/{num_test - fault_onset} = {100 * num_cusum_fault / (num_test - fault_onset):.1f}%")

    # 6. Detailed timeline
    print("\n6. Alarm timeline around fault onset:")
    print(f"   {'Sample':>7} | {'C_A':>8} | {'C_B':>8} | {'EWMA':>6} | {'CUSUM':>6}")
    print("   " + "-" * 42)

    spc.reset()  # Reset for replay
    for i in range(max(0, fault_onset - 5), min(num_test, fault_onset + 20)):
        result = spc.update(test_data[i])
        ewma_flag = "ALARM" if result["ewma_alarm"].any() else "ok"
        cusum_flag = "ALARM" if result["cusum_alarm"].any() else "ok"
        marker = " <-- FAULT" if i == fault_onset else ""
        print(
            f"   {i:>7} | {test_data[i, 0]:>8.4f} | {test_data[i, 1]:>8.4f} | "
            f"{ewma_flag:>6} | {cusum_flag:>6}{marker}"
        )

    print("\n" + "=" * 60)
    print("Example 12 complete!")
    print("Key insight: SPC charts (EWMA and CUSUM) detect small sustained")
    print("shifts that might be missed by simple threshold alarms. CUSUM is")
    print("especially effective for detecting gradual drift.")
    print("=" * 60)


if __name__ == "__main__":
    main()
