from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


TEMPERATURES: list[int] = [100, 200, 300, 400]

# Speed of light in cm / ps.
C_CM_PER_PS = 2.99792458e-2

# timestep between MD frames
DT_SAMPLE_PS = 10.0 * 0.001


def vacf_to_vdos_cm1(
    vacf: np.typing.NDArray,
    dt_sample_ps: float,
) -> tuple[np.typing.NDArray, np.typing.NDArray]:

    if vacf[0] <= 0.0:
        raise ValueError("C(0) must be positive.")

    # Make normalization explicit.
    vacf /= vacf[0]

    # TorchSim computes each VACF window using a zero-padded FFT of
    # length 2*N and returns lags 0, ..., N-1. Reconstruct the full,
    # even autocorrelation:
    #
    # [C(0), C(1), ..., C(N-1), 0, C(N-1), ..., C(1)]
    #
    # Its FFT is the corresponding nonnegative velocity periodogram.
    c_full = np.concatenate(
        (
            vacf,
            np.zeros(1),
            vacf[1:][::-1],
        )
    )

    # Two-sided power spectral density evaluated at nonnegative
    # frequencies. Multiplication by dt gives density per THz because
    # 1 / ps = THz.
    g_per_THz = np.fft.rfft(c_full).real * dt_sample_ps

    # Convert to a one-sided density. Do not double DC or Nyquist.
    if g_per_THz.size > 2:
        g_per_THz[1:-1] *= 2.0

    frequency_THz = np.fft.rfftfreq(
        c_full.size,
        d=dt_sample_ps,
    )

    # f [1/ps] / c [cm/ps] = wavenumber [1/cm].
    wavenumber_cm1 = frequency_THz / C_CM_PER_PS

    # Preserve the integrated spectral weight under the change of
    # variable from THz to cm^-1:
    #
    # g_cm1 d(wavenumber) = g_THz df
    #
    # df / d(wavenumber) = c.
    g_per_cm1 = g_per_THz * C_CM_PER_PS
    #g_per_cm1 = np.clip(g_per_cm1, a_min=0.0, a_max=None)

    return wavenumber_cm1, g_per_cm1


def main():

    metals = ["W", "Mo", "Ta"]
    chalcogen_pairs = [("S", "Te"), ("Se", "Te")]

    norm = mpl.colors.Normalize(
        vmin=min(TEMPERATURES),
        vmax=max(TEMPERATURES)
    )
    cmap = mpl.colormaps["cividis"]

    vacf_fig, vacf_axs = plt.subplots(
        ncols=2, nrows=len(metals) * len(chalcogen_pairs), sharex=True, sharey="row", figsize=[6.4, 9.6]
    )
    vdos_fig, vdos_axs = plt.subplots(
        ncols=2, nrows=len(metals) * len(chalcogen_pairs), sharex=True, sharey="row", figsize=[6.4, 9.6]
    )

    for subset_idx, subset in enumerate(["mace", "tce"]):

        vacf_axs[0, subset_idx].set_title(f"{subset.upper()}-selected subset")
        vdos_axs[0, subset_idx].set_title(f"{subset.upper()}-selected subset")

        for (vacf_ax, vdos_ax), (metal, chalcogen_pair) in zip(
            zip(vacf_axs[:, subset_idx].ravel(), vdos_axs[:, subset_idx].ravel(), strict=True),
            product(metals, chalcogen_pairs), 
            strict=True
        ):

            system_path = Path("md-data") / subset / f"{metal}{''.join(chalcogen_pair)}"
            vacf = np.loadtxt(system_path / "vacf.txt")

            y_offset = 1.0e-4
            for i, (temperature, vacf_temp) in enumerate(zip(TEMPERATURES, vacf, strict=True)):

                color = cmap(norm(temperature))

                md_time_fs = np.linspace(0, 100_000, len(vacf_temp))
                md_time_ps = md_time_fs / 1.0e+3
                vacf_ax.plot(md_time_ps, vacf_temp + i * 0.1, color=color)

                wavenumber, vdos = vacf_to_vdos_cm1(vacf_temp, dt_sample_ps=DT_SAMPLE_PS)
                mask = wavenumber < 400
                wavenumber = wavenumber[mask]
                vdos = vdos[mask]

                vdos_intensity_offset = vdos ** 2 + i * y_offset
                vdos_ax.plot(wavenumber, vdos_intensity_offset, color=color, zorder=6)

            for ax in vacf_ax, vdos_ax:
                if subset_idx == 0:
                    ax.set_ylabel(rf"$\text{{{metal}}}(\text{{{', '.join(chalcogen_pair)}}})_2$")
                ax.tick_params(labelleft=False)
                ax.grid()

    vdos_fig.supxlabel(r"wavenumber $\tilde{\nu}$ (cm$^{-1}$)")
    vdos_fig.supylabel(r"Intensity $|g(\omega)|^2$ (a.u.)")
    vdos_fig.tight_layout()

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(TEMPERATURES)

    cax = vdos_fig.add_axes([0.075, 0.025, 0.15, 0.01])
    cbar = vdos_fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"Temperature $T$ (K)")
    cbar.set_ticks(TEMPERATURES)
    vdos_fig.savefig("figures/spectra.pdf", bbox_inches="tight")
    plt.close(vdos_fig)

    vacf_fig.supxlabel(r"time $t$ (ps)")
    vacf_fig.supylabel(r"VACF $C(t)$ (a.u.)")
    vacf_fig.tight_layout()

    cax = vacf_fig.add_axes([0.075, 0.025, 0.15, 0.01])
    cbar = vacf_fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"Temperature $T$ (K)")
    cbar.set_ticks(TEMPERATURES)
    vacf_fig.savefig("figures/vacf.pdf", bbox_inches="tight")
    plt.close(vacf_fig)


if __name__ == "__main__":

    main()
