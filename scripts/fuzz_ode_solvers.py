#!/usr/bin/env python3
"""Fuzzing script for ReactorTwin ODE solvers.

Stress-tests all reactor types with random initial conditions,
parameter perturbations, and various time spans.
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Reactor imports
# ---------------------------------------------------------------------------
from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.batch import BatchReactor
from reactor_twin.reactors.semi_batch import SemiBatchReactor
from reactor_twin.reactors.pfr import PlugFlowReactor
from reactor_twin.reactors.fluidized_bed import FluidizedBedReactor
from reactor_twin.reactors.membrane import MembraneReactor
from reactor_twin.reactors.multi_phase import MultiPhaseReactor
from reactor_twin.reactors.population_balance import PopulationBalanceReactor
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SEEDS = 20
TIME_SPANS = [
    ("short", 0.01),
    ("medium", 1.0),
    ("long", 100.0),
]
NEGATIVE_TOL = -1e-6  # tolerance for "close to zero" negative concentrations


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class FuzzResult:
    reactor_type: str
    total: int = 0
    success: int = 0
    failure: int = 0
    nan_count: int = 0
    neg_conc_count: int = 0
    shape_errors: int = 0
    creation_errors: int = 0
    details: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------
def perturb_positive(value: float, rng: np.random.Generator, scale: float = 0.5) -> float:
    """Perturb a scalar value by +/- scale fraction, keep > 0."""
    factor = 1.0 + rng.uniform(-scale, scale)
    return max(abs(value * factor), 1e-12)


def perturb_array_positive(arr: list | np.ndarray, rng: np.random.Generator, scale: float = 0.5) -> list:
    """Perturb each element of an array, keep >= 0."""
    arr = np.array(arr, dtype=float)
    factors = 1.0 + rng.uniform(-scale, scale, size=arr.shape)
    result = np.maximum(arr * factors, 0.0)
    return result.tolist()


def perturb_temperature(T: float, rng: np.random.Generator, scale: float = 50.0) -> float:
    """Perturb temperature by +/- scale K, keep > 200 K."""
    return max(T + rng.uniform(-scale, scale), 200.0)


# ---------------------------------------------------------------------------
# Reactor factory functions
# ---------------------------------------------------------------------------
def make_cstr(rng: np.random.Generator) -> CSTRReactor:
    """Create a CSTR with perturbed parameters and Arrhenius kinetics."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(7.2e10, rng)]),
            "Ea": np.array([perturb_positive(7.27e4, rng)]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    params = {
        "V": perturb_positive(100.0, rng),
        "F": perturb_positive(100.0, rng),
        "C_feed": perturb_array_positive([1.0, 0.0], rng),
        "T_feed": perturb_temperature(350.0, rng),
        "C_initial": perturb_array_positive([0.5, 0.0], rng),
        "T_initial": perturb_temperature(350.0, rng),
        "rho": perturb_positive(1000.0, rng),
        "Cp": perturb_positive(0.239, rng),
        "UA": perturb_positive(5e4, rng),
        "T_coolant": perturb_temperature(300.0, rng),
        "dH_rxn": [-perturb_positive(5e4, rng)],
    }
    return CSTRReactor(
        name="fuzz_cstr",
        num_species=2,
        params=params,
        kinetics=kinetics,
        isothermal=False,
    )


def make_batch(rng: np.random.Generator) -> BatchReactor:
    """Create a Batch reactor with perturbed parameters."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(1e5, rng)]),
            "Ea": np.array([perturb_positive(5e4, rng)]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    params = {
        "V": perturb_positive(10.0, rng),
        "T": perturb_temperature(350.0, rng),
        "C_initial": perturb_array_positive([1.0, 0.0], rng),
    }
    return BatchReactor(
        name="fuzz_batch",
        num_species=2,
        params=params,
        kinetics=kinetics,
        isothermal=True,
        constant_volume=True,
    )


def make_semi_batch(rng: np.random.Generator) -> SemiBatchReactor:
    """Create a SemiBatch reactor with perturbed parameters."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(1e5, rng)]),
            "Ea": np.array([perturb_positive(5e4, rng)]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    params = {
        "V": perturb_positive(10.0, rng),
        "T": perturb_temperature(350.0, rng),
        "F_in": perturb_positive(1.0, rng),
        "C_in": perturb_array_positive([2.0, 0.0], rng),
        "C_initial": perturb_array_positive([0.5, 0.0], rng),
        "V_initial": perturb_positive(5.0, rng),
    }
    return SemiBatchReactor(
        name="fuzz_semi_batch",
        num_species=2,
        params=params,
        kinetics=kinetics,
        isothermal=True,
    )


def make_pfr(rng: np.random.Generator) -> PlugFlowReactor:
    """Create a PFR with perturbed parameters (small grid for speed)."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(1e3, rng)]),
            "Ea": np.array([perturb_positive(3e4, rng)]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    params = {
        "L": perturb_positive(2.0, rng),
        "u": perturb_positive(0.5, rng),
        "D": perturb_positive(0.01, rng),
        "C_in": perturb_array_positive([1.0, 0.0], rng),
        "T": perturb_temperature(400.0, rng),
    }
    return PlugFlowReactor(
        name="fuzz_pfr",
        num_species=2,
        params=params,
        kinetics=kinetics,
        num_cells=10,  # small for speed
    )


def make_fluidized_bed(rng: np.random.Generator) -> FluidizedBedReactor:
    """Create a FluidizedBed reactor with perturbed parameters."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(1e3, rng)]),
            "Ea": np.array([perturb_positive(3e4, rng)]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    # u_0 must be > u_mf
    u_mf = perturb_positive(0.05, rng)
    u_0 = u_mf + perturb_positive(0.1, rng)  # always > u_mf
    params = {
        "u_mf": u_mf,
        "u_0": u_0,
        "epsilon_mf": 0.4 + rng.uniform(-0.1, 0.1),  # keep in (0, 1)
        "d_b": perturb_positive(0.05, rng),
        "H_bed": perturb_positive(2.0, rng),
        "A_bed": perturb_positive(1.0, rng),
        "K_be": perturb_positive(2.0, rng),
        "C_feed": perturb_array_positive([1.0, 0.0], rng),
        "T_feed": perturb_temperature(400.0, rng),
    }
    return FluidizedBedReactor(
        name="fuzz_fluidized_bed",
        num_species=2,
        params=params,
        kinetics=kinetics,
        isothermal=True,
    )


def make_membrane(rng: np.random.Generator) -> MembraneReactor:
    """Create a Membrane reactor with perturbed parameters."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(1e3, rng)]),
            "Ea": np.array([perturb_positive(3e4, rng)]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    params = {
        "V_ret": perturb_positive(10.0, rng),
        "V_perm": perturb_positive(5.0, rng),
        "F_ret": perturb_positive(5.0, rng),
        "F_perm": perturb_positive(2.0, rng),
        "A_membrane": perturb_positive(0.5, rng),
        "Q": [perturb_positive(0.1, rng)],  # one permeating species
        "permeating_species_indices": [1],  # product B permeates
        "permeation_law": rng.choice(["linear", "sievert"]),
        "C_ret_feed": perturb_array_positive([1.0, 0.0], rng),
        "T_feed": perturb_temperature(400.0, rng),
    }
    return MembraneReactor(
        name="fuzz_membrane",
        num_species=2,
        params=params,
        kinetics=kinetics,
        isothermal=True,
    )


def make_multi_phase(rng: np.random.Generator) -> MultiPhaseReactor:
    """Create a MultiPhase (gas-liquid) reactor with perturbed parameters."""
    # 3 liquid species (A, B dissolved gas, C product), 1 gas species (B_gas)
    kinetics = ArrheniusKinetics(
        name="A_plus_B_to_C",
        num_reactions=1,
        params={
            "k0": np.array([perturb_positive(1e3, rng)]),
            "Ea": np.array([perturb_positive(3e4, rng)]),
            "stoich": np.array([[-1, -1, 1]]),
            "orders": np.array([[1, 1, 0]]),
        },
    )
    params = {
        "V_L": perturb_positive(10.0, rng),
        "V_G": perturb_positive(5.0, rng),
        "F_L": perturb_positive(2.0, rng),
        "F_G": perturb_positive(5.0, rng),
        "kLa": perturb_positive(5.0, rng),
        "H": [perturb_positive(10.0, rng)],  # Henry's constant for gas B
        "C_L_feed": perturb_array_positive([1.0, 0.0, 0.0], rng),
        "C_G_feed": perturb_array_positive([1.0], rng),
        "T_feed": perturb_temperature(350.0, rng),
        "gas_species_indices": [1],  # liquid species index 1 <-> gas species 0
    }
    return MultiPhaseReactor(
        name="fuzz_multi_phase",
        num_species=3,
        params=params,
        kinetics=kinetics,
        isothermal=True,
    )


def make_population_balance(rng: np.random.Generator) -> PopulationBalanceReactor:
    """Create a PopulationBalance (crystallizer) with perturbed parameters."""
    C_sat = perturb_positive(1.0, rng)
    params = {
        "V": perturb_positive(10.0, rng),
        "C_sat": C_sat,
        "kg": perturb_positive(1e-4, rng),
        "g": 1.0 + rng.uniform(0, 1.0),  # exponent 1-2
        "kb": perturb_positive(1e6, rng),
        "b": 1.0 + rng.uniform(0, 2.0),  # exponent 1-3
        "shape_factor": perturb_positive(0.5, rng),
        "rho_crystal": perturb_positive(2000.0, rng),
        "C_initial": C_sat * (1.0 + rng.uniform(0.1, 1.0)),  # supersaturated
    }
    return PopulationBalanceReactor(
        name="fuzz_pop_balance",
        num_species=1,
        params=params,
        isothermal=True,
        num_moments=4,
    )


# ---------------------------------------------------------------------------
# Map of reactor types to factory functions
# ---------------------------------------------------------------------------
REACTOR_FACTORIES = {
    "CSTR": make_cstr,
    "Batch": make_batch,
    "SemiBatch": make_semi_batch,
    "PFR": make_pfr,
    "FluidizedBed": make_fluidized_bed,
    "Membrane": make_membrane,
    "MultiPhase": make_multi_phase,
    "PopulationBalance": make_population_balance,
}


# ---------------------------------------------------------------------------
# Core fuzz runner
# ---------------------------------------------------------------------------
def fuzz_single(
    reactor_type: str,
    factory,
    seed: int,
    t_label: str,
    t_span_end: float,
    result: FuzzResult,
) -> None:
    """Run a single fuzz trial for one reactor / seed / time-span combo."""
    result.total += 1
    tag = f"{reactor_type} seed={seed} t={t_label}"
    rng = np.random.default_rng(seed)

    # --- 1. Create reactor ---
    try:
        reactor = factory(rng)
    except Exception as exc:
        result.creation_errors += 1
        result.failure += 1
        result.details.append(f"[CREATION ERROR] {tag}: {exc}")
        return

    # --- 2. Get initial state and perturb it ---
    y0 = reactor.get_initial_state().copy()
    # Perturb initial state: multiply by random factor in [0.5, 1.5], keep >= 0
    perturb = 1.0 + rng.uniform(-0.5, 0.5, size=y0.shape)
    y0 = np.maximum(y0 * perturb, 0.0)

    # For reactors with temperature as last state, ensure T stays reasonable
    if hasattr(reactor, "isothermal") and not getattr(reactor, "isothermal", True):
        y0[-1] = max(y0[-1], 250.0)

    # --- 3. Check ODE RHS shape ---
    try:
        rhs_out = reactor.ode_rhs(0.0, y0)
        if rhs_out.shape != y0.shape:
            result.shape_errors += 1
            result.failure += 1
            result.details.append(
                f"[SHAPE ERROR] {tag}: "
                f"ode_rhs shape {rhs_out.shape} != y0 shape {y0.shape}"
            )
            return
    except Exception as exc:
        result.shape_errors += 1
        result.failure += 1
        result.details.append(f"[RHS ERROR] {tag}: {exc}")
        return

    # --- 4. Integrate ---
    t_span = (0.0, t_span_end)
    t_eval = np.linspace(0.0, t_span_end, 50)

    try:
        sol = solve_ivp(
            lambda t, y: reactor.ode_rhs(t, y),
            t_span,
            y0,
            method="LSODA",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
            max_step=t_span_end / 10.0,
        )
    except Exception as exc:
        result.failure += 1
        result.details.append(f"[SOLVE ERROR] {tag}: {exc}")
        return

    # --- 5. Check success ---
    if not sol.success:
        result.failure += 1
        result.details.append(f"[SOLVER FAIL] {tag}: {sol.message}")
        return

    # --- 6. Check finite ---
    if not np.all(np.isfinite(sol.y)):
        result.nan_count += 1
        result.failure += 1
        nan_count = np.sum(~np.isfinite(sol.y))
        result.details.append(
            f"[NaN/Inf] {tag}: {nan_count} non-finite values in solution"
        )
        return

    # --- 7. Check non-negative concentrations ---
    # Concentration indices depend on reactor type
    n_species = reactor.num_species
    has_neg = False

    if isinstance(reactor, PopulationBalanceReactor):
        # State: [C, mu_0, ..., mu_{N-1}]  -- only check C (index 0)
        conc_vals = sol.y[0, :]
        if np.any(conc_vals < NEGATIVE_TOL):
            has_neg = True
    elif isinstance(reactor, PlugFlowReactor):
        # All states are concentrations
        if np.any(sol.y < NEGATIVE_TOL):
            has_neg = True
    elif isinstance(reactor, FluidizedBedReactor):
        # Bubble + emulsion concentrations (first 2*n_species)
        conc_vals = sol.y[: 2 * n_species, :]
        if np.any(conc_vals < NEGATIVE_TOL):
            has_neg = True
    elif isinstance(reactor, MembraneReactor):
        # Retentate (n_species) + permeate (num_permeating)
        n_total_conc = n_species + reactor.num_permeating
        conc_vals = sol.y[:n_total_conc, :]
        if np.any(conc_vals < NEGATIVE_TOL):
            has_neg = True
    elif isinstance(reactor, MultiPhaseReactor):
        # Liquid (n_species) + gas (num_gas_species)
        n_total_conc = n_species + reactor.num_gas_species
        conc_vals = sol.y[:n_total_conc, :]
        if np.any(conc_vals < NEGATIVE_TOL):
            has_neg = True
    elif isinstance(reactor, SemiBatchReactor):
        # [C_0, ..., C_n, V, (T)] -- check first n_species
        conc_vals = sol.y[:n_species, :]
        if np.any(conc_vals < NEGATIVE_TOL):
            has_neg = True
    else:
        # CSTR, Batch: first n_species are concentrations
        conc_vals = sol.y[:n_species, :]
        if np.any(conc_vals < NEGATIVE_TOL):
            has_neg = True

    if has_neg:
        result.neg_conc_count += 1
        min_val = float(np.min(conc_vals))
        result.details.append(
            f"[NEG CONC] {tag}: min concentration = {min_val:.6e}"
        )
        # Still count as success if solver converged -- negative is a physics
        # violation but solver did its job.  We track it separately.
        result.success += 1
        return

    # All checks passed
    result.success += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("  ReactorTwin ODE Solver Fuzzing Suite")
    print("=" * 78)
    print()
    print(f"  Seeds per reactor type : {NUM_SEEDS}")
    print(f"  Time spans             : {[t[0] for t in TIME_SPANS]}")
    print(f"  Total runs per type    : {NUM_SEEDS * len(TIME_SPANS)}")
    print(f"  Total runs             : {NUM_SEEDS * len(TIME_SPANS) * len(REACTOR_FACTORIES)}")
    print()

    results: dict[str, FuzzResult] = {}
    t_start = time.time()

    for rtype, factory in REACTOR_FACTORIES.items():
        res = FuzzResult(reactor_type=rtype)
        results[rtype] = res

        print(f"  Fuzzing {rtype:<22s} ", end="", flush=True)
        type_start = time.time()

        for seed in range(NUM_SEEDS):
            for t_label, t_end in TIME_SPANS:
                fuzz_single(rtype, factory, seed, t_label, t_end, res)

        elapsed = time.time() - type_start
        print(
            f"done  ({res.total:3d} runs, "
            f"{res.success:3d} ok, "
            f"{res.failure:3d} fail, "
            f"{elapsed:.1f}s)"
        )

    elapsed_total = time.time() - t_start
    print()

    # -----------------------------------------------------------------------
    # Print detailed errors (if any)
    # -----------------------------------------------------------------------
    any_details = any(r.details for r in results.values())
    if any_details:
        print("-" * 78)
        print("  Detailed Findings")
        print("-" * 78)
        for rtype, res in results.items():
            if res.details:
                print(f"\n  [{rtype}]")
                for d in res.details:
                    print(f"    {d}")
        print()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  Summary Table")
    print("=" * 78)
    header = (
        f"  {'Reactor Type':<22s} "
        f"{'Total':>6s} "
        f"{'OK':>6s} "
        f"{'Fail':>6s} "
        f"{'NaN':>6s} "
        f"{'NegC':>6s} "
        f"{'Shape':>6s} "
        f"{'Create':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_all = 0
    success_all = 0
    fail_all = 0

    for rtype, res in results.items():
        total_all += res.total
        success_all += res.success
        fail_all += res.failure
        print(
            f"  {res.reactor_type:<22s} "
            f"{res.total:6d} "
            f"{res.success:6d} "
            f"{res.failure:6d} "
            f"{res.nan_count:6d} "
            f"{res.neg_conc_count:6d} "
            f"{res.shape_errors:6d} "
            f"{res.creation_errors:6d}"
        )

    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'TOTAL':<22s} "
        f"{total_all:6d} "
        f"{success_all:6d} "
        f"{fail_all:6d}"
    )
    print()
    print(f"  Elapsed: {elapsed_total:.1f}s")
    print()

    if fail_all > 0:
        print(f"  RESULT: {fail_all} failure(s) detected.")
    else:
        print("  RESULT: All runs passed.")
    print()

    return 1 if fail_all > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
