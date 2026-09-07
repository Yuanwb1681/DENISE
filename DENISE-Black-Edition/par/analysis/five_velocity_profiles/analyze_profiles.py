from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NX, NY, DH = 500, 174, 20.0
X_POSITIONS_M = np.array([1000.0, 3000.0, 5000.0, 7000.0, 9000.0])

ROOT = Path("/home/ywb/DENISE-Black-Edition/par")
OUT = ROOT / "analysis/five_velocity_profiles"
SOURCES = {
    "initial": ROOT / "start/marmousi_II_start_1D.vp",
    "true": ROOT / "start/marmousi_II_marine.vp",
    "inverted": ROOT / "model/modelTest_vp_stage_4.bin",
}


def load_model(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    expected = NX * NY
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {values.size}")
    # DENISE files are written with x as the outer loop and y as the inner loop.
    return values.reshape(NX, NY)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(path) for name, path in SOURCES.items()}
    x_indices = np.rint(X_POSITIONS_M / DH).astype(int) - 1
    depth_m = (np.arange(NY) + 1) * DH

    rows = []
    metrics = []
    for x_m, ix in zip(X_POSITIONS_M, x_indices):
        initial = models["initial"][ix]
        truth = models["true"][ix]
        inverted = models["inverted"][ix]
        for iy, depth in enumerate(depth_m):
            rows.append(
                {
                    "x_m": x_m,
                    "depth_m": depth,
                    "initial_vp_m_s": initial[iy],
                    "true_vp_m_s": truth[iy],
                    "inverted_vp_m_s": inverted[iy],
                }
            )
        for label, estimate in (("initial", initial), ("inverted", inverted)):
            error = estimate - truth
            metrics.append(
                {
                    "x_m": x_m,
                    "model": label,
                    "rmse_m_s": np.sqrt(np.mean(error**2)),
                    "mae_m_s": np.mean(np.abs(error)),
                    "bias_m_s": np.mean(error),
                    "correlation": np.corrcoef(estimate, truth)[0, 1],
                }
            )

    profiles = pd.DataFrame(rows)
    summary = pd.DataFrame(metrics)
    profiles.to_csv(OUT / "five_vp_profiles.csv", index=False)
    summary.to_csv(OUT / "profile_error_summary.csv", index=False)

    fig, axes = plt.subplots(1, 5, figsize=(16, 7), sharey=True)
    for ax, x_m in zip(axes, X_POSITIONS_M):
        profile = profiles[profiles.x_m == x_m]
        ax.plot(profile.initial_vp_m_s, profile.depth_m, label="Initial", lw=1.5)
        ax.plot(profile.true_vp_m_s, profile.depth_m, label="True", lw=2.0)
        ax.plot(profile.inverted_vp_m_s, profile.depth_m, label="Inverted", lw=1.5)
        ax.set_title(f"x = {x_m/1000:.1f} km")
        ax.set_xlabel("Vp (m/s)")
        ax.grid(alpha=0.25)
        ax.invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Marmousi Vp: initial, true, and inverted vertical profiles")
    fig.tight_layout()
    fig.savefig(OUT / "five_vp_profiles.png", dpi=180)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
