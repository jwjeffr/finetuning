from pathlib import Path
import argparse

from ase import io
import numpy as np
import torch
from torch_sim.models.mace import MaceModel
import torch_sim as ts
from torch_sim.constraints import FixCom

from utils import PerSystemVACF


TEMPERATURES: list[int] = [100, 200, 300, 400]
# in ps
DT_MD: float = 0.001
N_STEPS: int = 100_000
TCE_DATA_PATH: Path = Path.home() / "ordered-tmd-tce" / "mc-data"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("system_name", type=str)
    parser.add_argument("subset_type", choices=["tce", "mace"])
    args = parser.parse_args()

    device = "cuda"

    potential_path = Path("potential-dirs") / f"{args.subset_type}-subset" / f"MACE_{args.subset_type}-subset.model"
    mace_model = MaceModel(model=potential_path, device=device)

    md_data_dir = Path("md-data") / args.subset_type
    system_dir = md_data_dir / args.system_name
    system_dir.mkdir(exist_ok=True, parents=True)

    alloys = [
        io.read(
            TCE_DATA_PATH / args.system_name / f"{temperature:0d}" / "frame_99.xyz",
            format="extxyz"
        )
        for temperature in TEMPERATURES
    ]

    # relax at 0 K
    systems = ts.optimize(
        system=alloys,
        model=mace_model,
        optimizer=ts.Optimizer.fire,
        autobatcher=True,
        init_kwargs=dict(cell_filter=ts.CellFilter.frechet)
    )

    # npt equilibrate
    systems = ts.integrate(
        system=systems,
        model=mace_model,
        n_steps=N_STEPS,
        timestep=DT_MD,
        temperature=TEMPERATURES,
        pbar=True,
        integrator=ts.Integrator.npt_langevin_isotropic,
        external_pressure=0.0
    )

    systems.constraints = [
        FixCom(torch.arange(systems.n_systems)).to(device=mace_model.device, dtype=mace_model.dtype)
    ]

    # nve run with VACFs
    systems.system_extras["vacf_id"] = torch.arange(
        systems.n_systems,
        device=systems.device,
        dtype=torch.long,
    )

    vacf_manager = PerSystemVACF(
        n_systems=systems.n_systems,
        window_size=2_000,
        device=systems.device,
        use_running_average=True,
        normalize=True,
    )

    sample_stride = 10
    reporter = ts.TrajectoryReporter(
        filenames=[
            system_dir / f".vacf_window_counter{temp:0d}.h5"
            for temp in TEMPERATURES
        ],
        state_frequency=100,
        prop_calculators={
            sample_stride: {
                "vacf_window_count": vacf_manager,
            }
        },
    )

    systems = ts.integrate(
        system=systems,
        model=mace_model,
        n_steps=N_STEPS,
        timestep=DT_MD,
        temperature=TEMPERATURES,
        integrator=ts.Integrator.nve,
        trajectory_reporter=reporter,
    )

    np.savetxt(
        system_dir / "vacf.txt", vacf_manager.vacfs.detach().cpu().numpy()
    )


if __name__ == "__main__":

    main()
