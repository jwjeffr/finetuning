from pathlib import Path

from ase import io
import numpy as np
from tce.calculator import TCECalculator
from mace.calculators import mace_mp
from e3nn import o3


CHALCOGENS: list[str] = ["S", "Se", "Te"]
TRANSITION_METALS: list[str] = ["Ta", "Mo", "W"]
SAMPLES_PATH: Path = Path("samples")


def main():

    alloys = []
    species_all = set()

    sample_paths = list(SAMPLES_PATH.rglob("relax"))
    sample_paths.sort(key=lambda path: int(path.parent.name))
    for path in sample_paths:

        alloy = io.read(path / "trajectory.xyz", format="extxyz", index=0)
        species_all |= set(alloy.symbols)

        positions = alloy.get_positions()

        order = np.lexsort((
            positions[:, 0],  # x
            positions[:, 1],  # y
            positions[:, 2],  # z
        ))

        alloy = alloy[order]
        alloys.append(alloy)

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
        species=sorted(species_all),
        neighbor_tolerance=0.05
    )

    X_tce = tce_calc.get_batched_feature_vectors(atoms_list=alloys)
    X_tce /= tce_calc.get_normalizer(alloys[0])

    mace_calc = mace_mp(model="large", device="cpu")
    irreps_out = o3.Irreps(str(mace_calc.models[0].products[0].linear.irreps_out))
    feature_length = irreps_out.dim // (irreps_out.lmax + 1) ** 2 * int(mace_calc.models[0].num_interactions)
    X_mace = np.zeros((len(alloys), feature_length))
    for sample, alloy in enumerate(alloys):
        print(f"MACE features: {(sample + 1) / len(alloys):.2%}", end="\r")
        X_mace[sample, :] = mace_calc.get_descriptors(atoms=alloy).mean(axis=0)
    print()

    np.savetxt(SAMPLES_PATH / "tce_features.txt", X_tce)
    np.savetxt(SAMPLES_PATH / "mace_features.txt", X_mace)


if __name__ == "__main__":

    main()

