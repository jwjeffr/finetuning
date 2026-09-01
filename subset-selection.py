from typing import Generator
from pathlib import Path

import numpy as np
from ase import io
import matplotlib.pyplot as plt


def maximum_entropy_subset_up_to_size_k(X: np.typing.NDArray, k: int) -> Generator[set[int], None, None]:

    X -= X.mean(axis=0)
    X = X.T

    d, n = X.shape

    subset = [np.argmin(np.linalg.norm(X, axis=0)).item()]
    mask = np.zeros(n, dtype=bool)
    mask[subset[0]] = True

    covariance = 1.0e-3 * np.eye(d)
    covariance += X[:, subset[0]:subset[0]+1] @ X[:, subset[0]:subset[0]+1].T

    while len(subset) <= k:

        yield set(subset)
        # Solve covariance * Y = X for all columns at once
        Y = np.linalg.solve(covariance, X)   # shape: (d, n)

        # Quadratic forms x_j^T covariance^{-1} x_j for all j at once
        quad = np.sum(X * Y, axis=0)         # shape: (n,)

        # Ignore already selected columns
        quad[mask] = 0.0

        gains = np.log1p(quad)
        j_new = int(np.argmax(gains))

        subset.append(j_new)
        mask[j_new] = True

        x_new = X[:, j_new:j_new+1]
        covariance += x_new @ x_new.T


def main():

    X_tce = np.loadtxt("samples/tce_features.txt")
    X_mace = np.loadtxt("samples/mace_features.txt")

    if len(X_tce) != len(X_mace):
        raise ValueError("dataset indices are different")
    n = len(X_tce)

    subset_sizes = []
    overlap_sizes = []

    max_subset_size = 30
    tce_subsets = maximum_entropy_subset_up_to_size_k(X_tce, k=max_subset_size)
    mace_subsets = maximum_entropy_subset_up_to_size_k(X_mace, k=max_subset_size)
    for tce_subset, mace_subset in zip(tce_subsets, mace_subsets, strict=True):

        assert (k := len(tce_subset)) == len(mace_subset)
        if k == 0:
            continue

        overlap = len(tce_subset & mace_subset)

        subset_sizes.append(k)
        overlap_sizes.append(overlap)
        print(f"Size {k:0d}: {overlap / k:.2%}")

    subset_sizes = np.array(subset_sizes)
    expected_overlaps = subset_sizes ** 2 / n
    plt.plot(subset_sizes, overlap_sizes, color="#FF5E5E", label="subset overlap")
    plt.plot(subset_sizes, expected_overlaps, color="black", linestyle="-.", label="random")
    plt.plot(subset_sizes, subset_sizes, color="black", linestyle="--", label="perfect overlap")
    plt.legend()
    plt.xlabel(r"Subset size $k = |\mathcal{X}_\text{TCE}| = |\mathcal{X}_\text{MACE}|$")
    plt.ylabel(r"Overlap size $|\mathcal{X}_\text{TCE}\cap \mathcal{X}_\text{MACE}|$")
    plt.grid()
    plt.tight_layout()

    figures_path = Path("figures")
    figures_path.mkdir(exist_ok=True)
    plt.savefig(figures_path / "subset-selection.pdf", bbox_inches="tight")

    all_samples = {}
    for pair_path in Path("samples").iterdir():

        if not pair_path.is_dir():
            continue

        relax = io.read(pair_path / "relax" / "trajectory.xyz", format="extxyz", index=slice(None))
        md = io.read(pair_path / "md" / "trajectory.xyz", format="extxyz", index=slice(None))

        index = int(pair_path.name)
        all_samples[index] = (relax, md)

    for trajectories in all_samples.values():
        for trajectory in trajectories:
            for frame in trajectory:
                frame.info["REF_energy"] = frame.get_potential_energy()
                frame.set_array("REF_forces", frame.get_forces())

    rng = np.random.default_rng(seed=0)
    remaining_trajectories = set(all_samples.keys()) - (tce_subset | mace_subset)
    test_subset = set(rng.choice(sorted(remaining_trajectories), size=10, replace=False))

    remaining_trajectories -= test_subset
    validation_subset = set(rng.choice(sorted(remaining_trajectories), size=5, replace=False))

    splits_path = Path("splits")
    splits_path.mkdir(exist_ok=True)

    names_and_sets = {"tce-subset": tce_subset, "mace-subset": mace_subset, "test": test_subset, "validation": validation_subset}

    for set1 in names_and_sets.values():
        for set2 in names_and_sets.values():
            if set1 == set2:
                continue
            if set1 == tce_subset and set2 == mace_subset:
                continue
            if set1 == mace_subset and set2 == tce_subset:
                continue
            assert set1 & set2 == set()

    for name, subset in names_and_sets.items():
        selected_trajectories = []
        for key, trajectory_pair in all_samples.items():
            if key not in subset:
                continue
            selected_trajectories.extend(trajectory_pair)

        io.write(
            splits_path / f"{name}.xyz", 
            sum(selected_trajectories, start=[]), 
            format="extxyz"
        )


if __name__ == "__main__":

    main()
