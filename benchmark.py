import argparse
from enum import StrEnum, auto
from pathlib import Path
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ase import io
import polars as pl
import numpy as np


SUBSETS = ["tce-subset", "mace-subset"]
SPLITS_DIR = Path("splits")
RESULTS_PATH = Path("benchmark_results.parquet")
PLOT_PATH = Path("figures") / "benchmarks"
RNG = np.random.default_rng(seed=0)


class ConfigType(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


CONFIG_TO_COLOR = {
    ConfigType.VALIDATION: "#FF5E5E",
    ConfigType.TEST: "#4CABFE",
    ConfigType.TRAIN: "#8FFF1E",
}


def make_grouped_axes(
    *,
    n_subsets: int,
    n_splits: int,
    figsize: tuple[float, float] = None,
):
    nrows = n_subsets * n_splits

    fig = plt.figure(figsize=figsize)

    outer_grid = fig.add_gridspec(
        nrows=n_subsets,
        ncols=1,
        hspace=0.6,  # spacing between subset groups
    )

    axs = np.empty((nrows, 2), dtype=object)

    shared_x = [None, None]
    shared_y = [None, None]

    for subset_index in range(n_subsets):
        inner_grid = outer_grid[subset_index].subgridspec(
            nrows=n_splits,
            ncols=2,
            hspace=0.25,  # spacing between train/validation/test
            wspace=0.25,
        )

        for split_index in range(n_splits):
            row_index = subset_index * n_splits + split_index

            for column_index in range(2):
                kwargs = {}

                if shared_x[column_index] is not None:
                    kwargs["sharex"] = shared_x[column_index]

                if shared_y[column_index] is not None:
                    kwargs["sharey"] = shared_y[column_index]

                ax = fig.add_subplot(
                    inner_grid[split_index, column_index],
                    **kwargs,
                )

                axs[row_index, column_index] = ax

                if shared_x[column_index] is None:
                    shared_x[column_index] = ax

                if shared_y[column_index] is None:
                    shared_y[column_index] = ax

    return fig, axs


def read_configs(path: Path):
    return io.read(
        path,
        format="extxyz",
        index=slice(None),
    )


def generate_results() -> list[dict]:

    from torch_sim.models.mace import MaceModel
    import torch
    import torch_sim as ts

    device = torch.device("cuda")

    validation_configs = read_configs(SPLITS_DIR / "validation.xyz")
    test_configs = read_configs(SPLITS_DIR / "test.xyz")

    results: list[dict] = []

    for subset_name in SUBSETS:
        train_configs = read_configs(
            SPLITS_DIR / f"{subset_name}.xyz"
        )

        # Each model sees the shared validation/test sets and only
        # its own training subset.
        config_groups = [
            (ConfigType.TRAIN, train_configs),
            (ConfigType.VALIDATION, validation_configs),
            (ConfigType.TEST, test_configs),
        ]

        configs = [
            config
            for _, group in config_groups
            for config in group
        ]

        config_types = [
            config_type
            for config_type, group in config_groups
            for _ in group
        ]
        
        model = MaceModel(
            model=f"potential-dirs/{subset_name}/MACE_{subset_name}.model",
            device=device
        )

        batch_results = ts.static(
            system=configs,
            model=model,
            autobatcher=True,
        )

        for config_index, (
            config_type,
            config,
            batch_result,
        ) in enumerate(
            zip(config_types, configs, batch_results, strict=True)
        ):
            actual_energy = float(config.get_potential_energy())
            actual_forces = config.get_forces()

            predicted_energy = float(
                batch_result["potential_energy"]
                .detach()
                .cpu()
                .item()
            )
            predicted_forces = (
                batch_result["forces"]
                .detach()
                .cpu()
                .tolist()
            )

            results.append(
                {
                    "model": subset_name,
                    "config_type": config_type.value,
                    "config_index": config_index,
                    "num_atoms": len(config),

                    "energy_actual_eV_per_atom": (
                        actual_energy / len(config)
                    ),
                    "energy_predicted_eV_per_atom": (
                        predicted_energy / len(config)
                    ),

                    "forces_actual_eV_per_A": (
                        actual_forces.tolist()
                    ),
                    "forces_predicted_eV_per_A": predicted_forces,
                }
            )

    return pl.DataFrame(results)


def save_results(results: pl.DataFrame, path: Path) -> None:
    results.write_parquet(
        path,
        compression="zstd",
    )


def load_results(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def plot_results(results: pl.DataFrame, path: Path) -> None:
    path.mkdir(exist_ok=True, parents=True)

    plot_order = (
        ConfigType.TRAIN,
        ConfigType.VALIDATION,
        ConfigType.TEST,
    )

    zorders = {
        ConfigType.TRAIN: 7,
        ConfigType.VALIDATION: 8,
        ConfigType.TEST: 9,
    }

    nrows = len(SUBSETS) * len(plot_order)

    n_splits = len(plot_order)
    nrows = len(SUBSETS) * n_splits

    fig_parity, axs_parity = make_grouped_axes(
        n_subsets=len(SUBSETS),
        n_splits=n_splits,
    )

    fig_residual, axs_residual = make_grouped_axes(
        n_subsets=len(SUBSETS),
        n_splits=n_splits,
    )

    for subset_index, subset_name in enumerate(SUBSETS):
        subset_results = results.filter(
            pl.col("model") == subset_name
        )

        for split_index, config_type in enumerate(plot_order):
            row_index = subset_index * len(plot_order) + split_index

            energy_ax_parity = axs_parity[row_index, 0]
            force_ax_parity = axs_parity[row_index, 1]

            energy_ax_residual = axs_residual[row_index, 0]
            force_ax_residual = axs_residual[row_index, 1]

            group = subset_results.filter(
                pl.col("config_type") == config_type.value
            )

            subset_label = subset_name.replace("-", " ")
            
            for ax in (
                energy_ax_parity,
                force_ax_parity,
                energy_ax_residual,
                force_ax_residual,
            ):
                if row_index % 3 == 0:
                    ax.set_title(subset_label)
                ax.grid()

            if group.height == 0:
                continue

            color = CONFIG_TO_COLOR[config_type]
            zorder = zorders[config_type]

            # Energy values
            actual_energies = group[
                "energy_actual_eV_per_atom"
            ].to_numpy()

            predicted_energies = group[
                "energy_predicted_eV_per_atom"
            ].to_numpy()

            energy_ax_residual.hist(
                1.0e3 * (predicted_energies - actual_energies),
                facecolor=color,
                edgecolor="black",
                zorder=zorder,
            )

            energy_sample_size = max(1, len(actual_energies) // 200)
            energy_indices = RNG.choice(
                len(actual_energies),
                size=min(energy_sample_size, len(actual_energies)),
                replace=False,
            )

            energy_ax_parity.scatter(
                actual_energies[energy_indices],
                predicted_energies[energy_indices],
                facecolor=color,
                edgecolor="black",
                zorder=zorder,
            )

            # Force values
            actual_forces = np.concatenate(
                [
                    np.asarray(force).ravel()
                    for force in group[
                        "forces_actual_eV_per_A"
                    ].to_list()
                ]
            )

            predicted_forces = np.concatenate(
                [
                    np.asarray(force).ravel()
                    for force in group[
                        "forces_predicted_eV_per_A"
                    ].to_list()
                ]
            )

            force_ax_residual.hist(
                predicted_forces - actual_forces,
                facecolor=color,
                edgecolor="black",
                zorder=zorder,
            )

            force_sample_size = max(1, len(actual_forces) // 10_000)
            force_indices = RNG.choice(
                len(actual_forces),
                size=min(force_sample_size, len(actual_forces)),
                replace=False,
            )

            force_ax_parity.scatter(
                actual_forces[force_indices],
                predicted_forces[force_indices],
                facecolor=color,
                edgecolor="black",
                zorder=zorder,
            )

            energy_ax_residual.set_yscale("log")
            force_ax_residual.set_yscale("log")

    # Axis labels
    for ax in axs_parity[:, 0][2::3]:
        ax.set_xlabel("DFT energy (eV/atom)")

    for ax in axs_parity[:, 1][2::3]:
        ax.set_xlabel("DFT force (eV/Å)")

    for ax in axs_residual[:, 0][2::3]:
        ax.set_xlabel("Energy error (meV/atom)")

    for ax in axs_residual[:, 1][2::3]:
        ax.set_xlabel("Force error (eV/Å)")

    for ax in chain(axs_residual.ravel(), axs_residual.ravel()):

        xlabel = ax.get_xlabel().lower()
        if ("force" in xlabel) or ("energy" in xlabel):
            continue

        plt.setp(ax.get_xticklabels(), visible=False)

    # Parity lines
    for ax in axs_parity.ravel():
        lower, upper = ax.get_xlim()
        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            color="black",
        )
        ax.set_xlim([lower, upper])

    legend_handles = [
        Patch(
            facecolor=CONFIG_TO_COLOR[config_type],
            edgecolor="black",
            label=config_type.value.lower(),
        )
        for config_type in plot_order
    ]

    fig_parity.legend(
        handles=legend_handles,
        loc="upper center",
        ncols=3,
    )
    fig_parity.supylabel("MACE energy or force (eV/atom or eV/Å)", x=0.0)
    fig_parity.savefig(
        path / "parity.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_parity)

    fig_residual.legend(
        handles=legend_handles,
        loc="upper center",
        ncols=3,
    )
    fig_residual.supylabel("counts", x=0.0)
    fig_residual.savefig(
        path / "residual.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig_residual)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["generate", "plot", "all"],
    )
    args = parser.parse_args()

    if args.command in {"generate", "all"}:
        results = generate_results()
        save_results(results, RESULTS_PATH)
        print(f"Saved {results.height} rows to {RESULTS_PATH}")

    if args.command in {"plot", "all"}:
        results = load_results(RESULTS_PATH)
        plot_results(results, PLOT_PATH)
        print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":

    main()
