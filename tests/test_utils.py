"""Tests for registry system and constants."""

from __future__ import annotations

import pytest

from reactor_twin.utils import (
    CONSTRAINT_REGISTRY,
    KINETICS_REGISTRY,
    NEURAL_DE_REGISTRY,
    R_GAS,
    REACTOR_REGISTRY,
    Registry,
)

# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the plugin registry system."""

    def test_register_and_get(self):
        """Register a class and retrieve it by key."""
        reg = Registry("test")

        @reg.register("my_class")
        class MyClass:
            pass

        assert reg.get("my_class") is MyClass

    def test_get_missing_key_raises(self):
        from reactor_twin.exceptions import RegistryError

        reg = Registry("test")
        with pytest.raises(RegistryError, match="not found"):
            reg.get("nonexistent")

    def test_list_keys(self):
        reg = Registry("test")

        @reg.register("alpha")
        class A:
            pass

        @reg.register("beta")
        class B:
            pass

        keys = reg.list_keys()
        assert "alpha" in keys
        assert "beta" in keys
        # list_keys returns sorted
        assert keys == sorted(keys)

    def test_contains(self):
        reg = Registry("test")

        @reg.register("exists")
        class X:
            pass

        assert "exists" in reg
        assert "nope" not in reg

    def test_repr(self):
        reg = Registry("test_repr")

        @reg.register("foo")
        class Foo:
            pass

        repr_str = repr(reg)
        assert "test_repr" in repr_str
        assert "foo" in repr_str

    def test_overwrite_warning(self):
        """Registering the same key twice should overwrite (with a warning)."""
        reg = Registry("test")

        @reg.register("same_key")
        class First:
            pass

        @reg.register("same_key")
        class Second:
            pass

        # Should return the latest registration
        assert reg.get("same_key") is Second

    def test_register_returns_original_class(self):
        """Decorator should return the original class unchanged."""
        reg = Registry("test")

        @reg.register("cls")
        class Original:
            value = 42

        assert Original.value == 42


# ---------------------------------------------------------------------------
# Constants Tests
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for physical constants."""

    def test_r_gas_value(self):
        """R_GAS should be the universal gas constant in J/(mol*K)."""
        assert R_GAS == pytest.approx(8.314, abs=0.001)

    def test_r_gas_type(self):
        assert isinstance(R_GAS, float)


# ---------------------------------------------------------------------------
# Global Registry Tests
# ---------------------------------------------------------------------------


class TestGlobalRegistries:
    """Tests that global registries contain expected entries after imports."""

    def test_reactor_registry_has_cstr(self):
        assert "cstr" in REACTOR_REGISTRY

    def test_reactor_registry_has_batch(self):
        assert "batch" in REACTOR_REGISTRY

    def test_reactor_registry_has_pfr(self):
        assert "pfr" in REACTOR_REGISTRY

    def test_reactor_registry_has_semi_batch(self):
        assert "semi_batch" in REACTOR_REGISTRY

    def test_kinetics_registry_has_arrhenius(self):
        assert "arrhenius" in KINETICS_REGISTRY

    def test_kinetics_registry_has_michaelis_menten(self):
        assert "michaelis_menten" in KINETICS_REGISTRY

    def test_kinetics_registry_has_power_law(self):
        assert "power_law" in KINETICS_REGISTRY

    def test_kinetics_registry_has_langmuir_hinshelwood(self):
        assert "langmuir_hinshelwood" in KINETICS_REGISTRY

    def test_kinetics_registry_has_reversible(self):
        assert "reversible" in KINETICS_REGISTRY

    def test_kinetics_registry_has_monod(self):
        assert "monod" in KINETICS_REGISTRY

    def test_constraint_registry_has_positivity(self):
        assert "positivity" in CONSTRAINT_REGISTRY

    def test_constraint_registry_has_mass_balance(self):
        assert "mass_balance" in CONSTRAINT_REGISTRY

    def test_neural_de_registry_has_neural_ode(self):
        assert "neural_ode" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_augmented(self):
        assert "augmented_neural_ode" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_bayesian(self):
        assert "bayesian_neural_ode" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_hybrid(self):
        assert "hybrid_neural_ode" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_foundation(self):
        assert "foundation_neural_ode" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_latent(self):
        assert "latent_neural_ode" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_sde(self):
        assert "neural_sde" in NEURAL_DE_REGISTRY

    def test_neural_de_registry_has_cde(self):
        assert "neural_cde" in NEURAL_DE_REGISTRY

    def test_reactor_registry_has_membrane(self):
        assert "membrane" in REACTOR_REGISTRY

    def test_reactor_registry_has_fluidized_bed(self):
        assert "fluidized_bed" in REACTOR_REGISTRY

    def test_reactor_registry_get_cstr_returns_class(self):
        from reactor_twin.reactors import CSTRReactor

        cls = REACTOR_REGISTRY.get("cstr")
        assert cls is CSTRReactor

    def test_kinetics_registry_get_arrhenius_returns_class(self):
        from reactor_twin.reactors.kinetics import ArrheniusKinetics

        cls = KINETICS_REGISTRY.get("arrhenius")
        assert cls is ArrheniusKinetics
