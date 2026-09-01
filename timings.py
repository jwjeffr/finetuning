from time import perf_counter_ns
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


NUM_SAMPLES: int = 15
RNG: np.random.Generator = np.random.default_rng(seed=0)
CHALCOGENS: list[str] = ["S", "Se", "Te"]
TRANSITION_METALS: list[str] = ["Ta", "Mo", "W"]
TIMINGS_PATH: Path = Path("timings.csv")


def generate_timings():

    from ase.build import mx2
    from ase.data import atomic_numbers
    from tce.calculator import TCECalculator
    from mace.calculators import mace_mp
    from e3nn import o3

    atomic_numbers_transition_metals = np.fromiter(
        (atomic_numbers[symbol] for symbol in TRANSITION_METALS),
        dtype=int
    )
    atomic_numbers_chalcogens = np.fromiter(
        (atomic_numbers[symbol] for symbol in CHALCOGENS),
        dtype=int
    )

    sheet = mx2(formula="WS2", size=(4, 4, 1), vacuum=40.0)
    sheet.pbc = True

    transition_metal_indices, _ = np.where(
        sheet.numbers[:, None] == atomic_numbers_transition_metals[None, :]
    )
    chalcogen_indices, _ = np.where(
        sheet.numbers[:, None] == atomic_numbers_chalcogens[None, :]
    )

    alloys = []
    for sample in range(NUM_SAMPLES):

        alloy = sheet.copy()

        # randomly choose a subset of transition metals and chalcogens
        num_transition_metals = RNG.integers(low=1, high=len(TRANSITION_METALS), endpoint=True)
        num_chalcogens = RNG.integers(low=1, high=len(CHALCOGENS), endpoint=True)

        transition_metals_sampled = RNG.choice(TRANSITION_METALS, replace=False, size=num_transition_metals)
        chalcogens_sampled = RNG.choice(CHALCOGENS, replace=False, size=num_chalcogens)

        metal_composition = RNG.uniform(low=0.0, high=1.0, size=len(transition_metals_sampled))
        metal_composition /= metal_composition.sum()
        chalcogen_composition = RNG.uniform(low=0.0, high=1.0, size=len(chalcogens_sampled))
        chalcogen_composition /= chalcogen_composition.sum()

        alloy.symbols[transition_metal_indices] = RNG.choice(
            a=transition_metals_sampled,
            p=metal_composition,
            size=len(transition_metal_indices)
        )
        alloy.symbols[chalcogen_indices] = RNG.choice(
            a=chalcogens_sampled,
            p=chalcogen_composition,
            size=len(chalcogen_indices)
        )
        
        alloys.append(alloy)

    subset_sizes = np.arange(1, NUM_SAMPLES)
    tce_timings_means = []
    tce_timings_stds = []
    mace_timings_means = []
    mace_timings_stds = []

    with open(TIMINGS_PATH, "w") as file:
        print("subset_size, tce_mean, tce_std, mace_mean, mace_std", file=file)
    
    for subset_size in subset_sizes:

        subset = RNG.choice(len(alloys), size=subset_size, replace=False)
        subset = [alloys[i] for i in subset]

        tce_timings = []
        mace_timings = []

        for _ in range(5):

            # reinitialize calculators to make it fair
            # tce has a cache-ing mechanism
            tce_calc = TCECalculator(
                neighbor_cutoffs=[2.5, 3.25, 4.0, 4.5, 5.0, 5.5],
                many_body_features=[
                    (0, 0, 1),
                    (1, 1, 1),
                    (0, 1, 2),
                    (1, 2, 2),
                    (0, 0, 0, 0, 1, 1),
                    (0, 0, 0, 1, 1, 1),
                    (0, 0, 1, 1, 1, 2)
                    
                ],
                species=TRANSITION_METALS + CHALCOGENS,
                neighbor_tolerance=0.05
            )
            past = perf_counter_ns()
            X_tce = tce_calc.get_batched_feature_vectors(atoms_list=subset)
            X_tce /= tce_calc.get_normalizer(alloys[0])
            tce_timings.append(perf_counter_ns() - past)

            mace_calc = mace_mp(model="large", device="cpu")
            irreps_out = o3.Irreps(str(mace_calc.models[0].products[0].linear.irreps_out))
            mace_feature_length = irreps_out.dim // (irreps_out.lmax + 1) ** 2 * int(mace_calc.models[0].num_interactions)
            past = perf_counter_ns()
            X_mace = np.zeros((NUM_SAMPLES, mace_feature_length))
            for sample, alloy in enumerate(subset):
                X_mace[sample, :] = mace_calc.get_descriptors(atoms=alloy).mean(axis=0)
            mace_timings.append(perf_counter_ns() - past)

        tce_timings_means.append(np.mean(tce_timings))
        tce_timings_stds.append(np.std(tce_timings))
        mace_timings_means.append(np.mean(mace_timings))
        mace_timings_stds.append(np.std(mace_timings))

        with open(TIMINGS_PATH, "a") as file:
            print(
                f"{subset_size:.0f}, {np.mean(tce_timings):.2f}, {np.std(tce_timings):.2f}, {np.mean(mace_timings):.2f}, {np.std(mace_timings):.2f}",
                file=file
            )


def plot_timings():

    tce_color = "#FF5E5E"
    mace_color = "#4CABFE"

    data = np.loadtxt(TIMINGS_PATH, delimiter=",", skiprows=1)
    data = data[1:].T

    sizes = data[0]
    timings = data[1:]

    timings /= 1.0e+9
    tce_mean, tce_std, mace_mean, mace_std = timings

    tce_reg = linregress(sizes, tce_mean)
    mace_reg = linregress(sizes, mace_mean)

    plt.errorbar(sizes, tce_mean, yerr=3.0 * tce_std, capsize=3.0, label=r"TCE $\pm 3\sigma$", fmt="o", mec="black", color=tce_color)
    plt.plot(sizes, tce_reg.intercept + tce_reg.slope * sizes, linestyle="--", color=tce_color, label=f"slope = {tce_reg.slope:.2f} sec")
    plt.errorbar(sizes, mace_mean, yerr=3.0 * mace_std, capsize=3.0, label=r"MACE $\pm 3\sigma$", fmt="s", mec="black", color=mace_color)
    plt.plot(sizes, mace_reg.intercept + mace_reg.slope * sizes, linestyle="--", color=mace_color, label=f"slope = {mace_reg.slope:.2f} sec")
    plt.grid()
    plt.legend()
    plt.xlabel("subset size $k$")
    plt.ylabel("computation time (s)")
    plt.tight_layout()
    plt.savefig(Path("figures") / "timings.pdf", bbox_inches="tight")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "plot"])

    args = parser.parse_args()
    if args.mode == "generate":
        generate_timings()
    if args.mode == "plot":
        plot_timings()


if __name__ == "__main__":

    main()
