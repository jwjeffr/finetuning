from dataclasses import dataclass

import torch
import torch_sim as ts
from torch_sim.properties.correlations import VelocityAutoCorrelation
from torch_sim.state import SimState
from torch_sim.models.interface import ModelInterface
from torch_sim.integrators.md import MDState
from torch_sim.monte_carlo import SwapMCState


# Speed of light in cm / ps.
C_CM_PER_PS = 2.99792458e-2


@dataclass(kw_only=True)
class HybridSwapMCState(SwapMCState, MDState):
    """State for hybrid MD-Monte Carlo simulations.

    This state class extends the standard MDState with:
    - last_permutation: Tracks whether the last MC move was accepted

    All other MD attributes (positions, momenta, forces, etc.) are inherited
    from MDState.
    """

    _atom_attributes = (
        ts.SwapMCState._atom_attributes | MDState._atom_attributes  # noqa: SLF001
    )
    _system_attributes = (
        ts.SwapMCState._system_attributes | MDState._system_attributes  # noqa: SLF001
    )

    def __post_init__(self) -> None:
        """Initialize HybridSwapMCState and ensure last_permutation is set."""
        super().__post_init__()


@torch.no_grad()
def vacf_to_vdos_cm1(
    vacf: torch.Tensor,
    dt_sample_ps: float,
    *,
    integral: float | None = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a TorchSim-native VACF into a one-sided VDOS versus
    spectroscopic wavenumber.

    Parameters
    ----------
    vacf
        One-dimensional VACF tensor with lags
        [0, dt_sample_ps, 2*dt_sample_ps, ...].

        This function is designed for the VACF returned by TorchSim's
        native CorrelationCalculator, including the mass-weighted
        subclass discussed above.

    dt_sample_ps
        Time between VACF samples in picoseconds. This is the MD time
        step times the VACF reporting stride.

    integral
        Desired integral of g over wavenumber.

        integral=1.0:
            Unit-normalized spectral distribution.

        integral=3*N - 3:
            VDOS normalized to the number of non-translational modes.

        integral=None:
            Retain the normalization implied by C(0).

    Returns
    -------
    wavenumber_cm1
        Spectroscopic wavenumber in cm^-1.

    g_per_cm1
        One-sided VDOS in inverse cm^-1. If integral=3*N-3, the units
        are states per cm^-1.
    """
    if vacf is None:
        raise ValueError("No complete VACF window has been collected.")

    if vacf.ndim != 1:
        raise ValueError("vacf must be one-dimensional.")

    if vacf.numel() < 2:
        raise ValueError("vacf must contain at least two time lags.")

    if dt_sample_ps <= 0.0:
        raise ValueError("dt_sample_ps must be positive.")

    # The FFT is tiny compared with the MD calculation, so use float64.
    c = vacf.detach().to(dtype=torch.float64)

    if not torch.isfinite(c).all():
        raise ValueError("vacf contains non-finite values.")

    if c[0] <= 0.0:
        raise ValueError("C(0) must be positive.")

    # Make normalization explicit.
    c = c / c[0]

    n_lags = c.numel()

    # TorchSim computes each VACF window using a zero-padded FFT of
    # length 2*N and returns lags 0, ..., N-1. Reconstruct the full,
    # even autocorrelation:
    #
    # [C(0), C(1), ..., C(N-1), 0, C(N-1), ..., C(1)]
    #
    # Its FFT is the corresponding nonnegative velocity periodogram.
    c_full = torch.cat(
        (
            c,
            c.new_zeros(1),
            c[1:].flip(0),
        )
    )

    # Two-sided power spectral density evaluated at nonnegative
    # frequencies. Multiplication by dt gives density per THz because
    # 1 / ps = THz.
    g_per_THz = (
        torch.fft.rfft(c_full).real
        * dt_sample_ps
    )

    # Convert to a one-sided density. Do not double DC or Nyquist.
    if g_per_THz.numel() > 2:
        g_per_THz[1:-1] *= 2.0

    frequency_THz = torch.fft.rfftfreq(
        c_full.numel(),
        d=dt_sample_ps,
    ).to(
        device=c.device,
        dtype=c.dtype,
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

    # For a VACF coming directly from TorchSim's biased finite-window
    # autocorrelation estimator, negative values should be roundoff
    # only. Large negative bins indicate that the VACF was modified,
    # differently windowed, or is otherwise inconsistent.
    """scale = g_per_cm1.abs().amax()
    tolerance = (
        100.0
        * torch.finfo(g_per_cm1.dtype).eps
        * scale
    )

    print(g_per_cm1)

    if g_per_cm1.min() < -tolerance:
        raise ValueError(
            "The transformed VACF has significant negative spectral "
            "weight. Do not truncate or independently window the "
            "TorchSim VACF before calling this function."
        )"""

    g_per_cm1 = g_per_cm1.clamp_min(0.0)

    if integral is not None:
        area = torch.trapezoid(
            g_per_cm1,
            wavenumber_cm1,
        )

        if area <= 0.0:
            raise ValueError("The VDOS has non-positive area.")

        g_per_cm1 *= float(integral) / area

    return wavenumber_cm1, g_per_cm1


class PerSystemVACF:
    """Maintain one independent VACF calculator per simulated system."""

    def __init__(
        self,
        n_systems: int,
        *,
        window_size: int,
        device: torch.device,
        use_running_average: bool = True,
        normalize: bool = True,
    ) -> None:
        
        self.calculators = [
            MassWeightedVelocityAutoCorrelation(
                window_size=window_size,
                device=device,
                use_running_average=use_running_average,
                normalize=normalize,
            )
            for _ in range(n_systems)
        ]

    def __call__(self, state, model=None) -> torch.Tensor:
        # `vacf_id` is preserved when the batched state is split.
        system_id = int(state.vacf_id.item())

        # This updates only the calculator belonging to this system.
        # The returned tensor is TorchSim's window count.
        return self.calculators[system_id](state, model)

    @property
    def vacfs(self) -> torch.Tensor:
        """Shape: (n_systems, window_size)."""
        return torch.stack(
            [calculator.vacf for calculator in self.calculators],
            dim=0,
        )


class MassWeightedVelocityAutoCorrelation(VelocityAutoCorrelation):

    r"""
    Mass-weighted VACF:

        C(t) =
            sum_i m_i <v_i(0) . v_i(t)>
            --------------------------------
            sum_i m_i <v_i(0) . v_i(0)>

    The native TorchSim windowing, FFT calculation, and running
    average are otherwise retained.
    """

    def __init__(
        self,
        *,
        window_size: int,
        device: torch.device,
        use_running_average: bool = True,
        normalize: bool = True,
    ) -> None:
        # Critical: do not normalize each atom/component separately,
        # because that would remove the mass weighting.
        super().__init__(
            window_size=window_size,
            device=device,
            use_running_average=use_running_average,
            normalize=False,
        )

        self._normalize_weighted = normalize

        # ACF[sqrt(m) v] = m * ACF[v].
        self.corr_calc.properties["velocity"] = (
            self._mass_weighted_velocity
        )


    @staticmethod
    def _mass_weighted_velocity(state: SimState) -> torch.Tensor:
        velocities = state.velocities
        sqrt_mass = torch.sqrt(
            state.masses.to(
                device=velocities.device,
                dtype=velocities.dtype,
            )
        )
        return velocities * sqrt_mass[:, None]


    def __call__(
        self,
        state: SimState,
        model: ModelInterface = None,
    ) -> torch.Tensor:
        
        # Native _avg is initially created with PyTorch's default
        # floating dtype. Match it to the MD state.

        if self._avg.dtype != state.velocities.dtype:
            self._avg = self._avg.to(
                dtype=state.velocities.dtype
            )

        return super().__call__(state, model)


    @property
    def vacf(self) -> torch.Tensor | None:
        # No complete correlation window has been collected yet.
        if self._window_count == 0:
            return None

        result = self._avg

        if not self._normalize_weighted:
            return result

        denominator = result[0].clamp_min(
            torch.finfo(result.dtype).tiny
        )
        return result / denominator
