"""Sensitivity analysis for reactor parameters using SALib."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.base import AbstractReactor

logger = logging.getLogger(__name__)

try:
    from SALib.analyze import morris as morris_analyze
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import morris as morris_sample
    from SALib.sample import sobol as saltelli

    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logger.warning(
        "SALib not installed. Sensitivity analysis unavailable. "
        "Install with: pip install reactor-twin[analysis]"
    )


class SensitivityAnalyzer:
    """Variance-based sensitivity analysis for reactor parameters.

    Wraps SALib to perform Sobol or Morris sensitivity analysis on
    how reactor parameters affect simulation outputs.

    Attributes:
        reactor: Reactor instance.
        param_names: Names of parameters to analyse.
        param_bounds: Lower/upper bounds for each parameter.
    """

    def __init__(
        self,
        reactor: AbstractReactor,
        param_names: list[str],
        param_bounds: list[list[float]],
    ):
        if not SALIB_AVAILABLE:
            raise ImportError(
                "SALib is required for sensitivity analysis. "
                "Install with: pip install reactor-twin[analysis]"
            )

        self.reactor = reactor
        self.param_names = param_names
        self.param_bounds = param_bounds

        self.problem: dict[str, Any] = {
            "num_vars": len(param_names),
            "names": param_names,
            "bounds": param_bounds,
        }

        logger.info(f"SensitivityAnalyzer: {len(param_names)} parameters, reactor={reactor.name}")

    def _evaluate_model(
        self,
        param_values: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray,
        output_index: int = 0,
    ) -> np.ndarray:
        """Run reactor simulation for each parameter set.

        Args:
            param_values: Parameter sample matrix, shape (N, num_params).
            t_span: Integration time interval.
            t_eval: Evaluation time points.
            output_index: Which state variable to use as output.

        Returns:
            Model outputs, shape (N,).
        """
        from scipy.integrate import solve_ivp

        # Validate output_index
        if output_index >= self.reactor.state_dim:
            raise ValueError(
                f"output_index={output_index} out of bounds for "
                f"reactor with state_dim={self.reactor.state_dim}"
            )

        outputs = np.zeros(param_values.shape[0])

        for i in range(param_values.shape[0]):
            # Deep copy reactor params for safe backup
            original_params = copy.deepcopy(self.reactor.params)
            for j, name in enumerate(self.param_names):
                self.reactor.params[name] = param_values[i, j]

            try:
                y0 = self.reactor.get_initial_state()
                sol = solve_ivp(
                    self.reactor.ode_rhs,
                    t_span,
                    y0,
                    t_eval=t_eval,
                    method="LSODA",
                )
                if sol.success:
                    outputs[i] = sol.y[output_index, -1]  # Final value
                else:
                    outputs[i] = np.nan
            except Exception:
                outputs[i] = np.nan
            finally:
                self.reactor.params = original_params

        # Handle NaN results
        valid_mask = np.isfinite(outputs)
        nan_fraction = 1.0 - valid_mask.mean() if len(outputs) > 0 else 0.0

        if nan_fraction > 0.5:
            raise RuntimeError(
                f"More than 50% of sensitivity evaluations failed "
                f"({nan_fraction:.0%} NaN). Check reactor configuration."
            )

        if valid_mask.any() and not valid_mask.all():
            logger.warning(
                f"Sensitivity analysis: {(~valid_mask).sum()}/{len(outputs)} "
                "evaluations returned NaN, replacing with median of valid results."
            )
            outputs[~valid_mask] = np.nanmedian(outputs)

        return outputs

    def analyze(
        self,
        t_span: tuple[float, float],
        t_eval: np.ndarray,
        method: str = "sobol",
        output_index: int = 0,
        N: int = 256,
    ) -> dict[str, Any]:
        """Run sensitivity analysis.

        Args:
            t_span: Integration time interval.
            t_eval: Evaluation time points.
            method: 'sobol' or 'morris'.
            output_index: State variable index to analyse.
            N: Base sample size (total samples = N*(2D+2) for Sobol).

        Returns:
            Dict with sensitivity indices (S1, ST, S2 for Sobol;
            mu, mu_star, sigma for Morris).
        """
        if method == "sobol":
            param_values = saltelli.sample(self.problem, N)
            outputs = self._evaluate_model(param_values, t_span, t_eval, output_index)
            result = sobol_analyze.analyze(self.problem, outputs)
            return {
                "S1": result["S1"],
                "ST": result["ST"],
                "S2": result.get("S2"),
                "method": "sobol",
                "param_names": self.param_names,
            }
        elif method == "morris":
            param_values = morris_sample.sample(self.problem, N)
            outputs = self._evaluate_model(param_values, t_span, t_eval, output_index)
            result = morris_analyze.analyze(self.problem, param_values, outputs)
            return {
                "mu": result["mu"],
                "mu_star": result["mu_star"],
                "sigma": result["sigma"],
                "method": "morris",
                "param_names": self.param_names,
            }
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sobol' or 'morris'.")


__all__ = ["SensitivityAnalyzer"]
