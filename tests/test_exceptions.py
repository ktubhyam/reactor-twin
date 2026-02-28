"""Tests for the ReactorTwin exception hierarchy."""

from __future__ import annotations

import pytest

from reactor_twin.exceptions import (
    ConfigurationError,
    ConstraintViolationError,
    ExportError,
    ReactorTwinError,
    RegistryError,
    SolverError,
    ValidationError,
)


# ── Inheritance chain ────────────────────────────────────────────────


class TestInheritance:
    """All custom exceptions inherit from ReactorTwinError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            SolverError,
            ValidationError,
            ExportError,
            ConstraintViolationError,
            RegistryError,
            ConfigurationError,
        ],
    )
    def test_subclass_of_base(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, ReactorTwinError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ReactorTwinError,
            SolverError,
            ValidationError,
            ExportError,
            ConstraintViolationError,
            RegistryError,
            ConfigurationError,
        ],
    )
    def test_subclass_of_exception(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, Exception)

    def test_base_not_subclass_of_subtypes(self) -> None:
        assert not issubclass(ReactorTwinError, SolverError)


# ── Error message propagation ────────────────────────────────────────


class TestMessagePropagation:
    """Exception messages round-trip correctly."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ReactorTwinError,
            SolverError,
            ValidationError,
            ExportError,
            ConstraintViolationError,
            RegistryError,
            ConfigurationError,
        ],
    )
    def test_message_preserved(self, exc_cls: type) -> None:
        msg = f"test message for {exc_cls.__name__}"
        exc = exc_cls(msg)
        assert str(exc) == msg

    def test_catch_base_catches_subtype(self) -> None:
        with pytest.raises(ReactorTwinError):
            raise SolverError("solver blew up")

    def test_catch_specific_does_not_catch_sibling(self) -> None:
        with pytest.raises(SolverError):
            raise SolverError("fail")
        # SolverError should not catch ValidationError
        with pytest.raises(ValidationError):
            raise ValidationError("bad input")


# ── Integration: existing code paths raise correct new types ─────────


class TestExistingCodePaths:
    """Verify migrated code raises the new exception types."""

    def test_registry_get_unknown_raises_registry_error(self) -> None:
        from reactor_twin.utils.registry import REACTOR_REGISTRY

        with pytest.raises(RegistryError, match="not found"):
            REACTOR_REGISTRY.get("nonexistent_reactor_xyz")

    def test_cstr_missing_param_raises_configuration_error(self) -> None:
        from reactor_twin.reactors.cstr import CSTRReactor

        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            CSTRReactor(
                name="bad",
                num_species=2,
                params={"F": 10.0, "C_feed": [1.0, 0.0], "T_feed": 350.0},
                isothermal=True,
            )

    def test_batch_missing_param_raises_configuration_error(self) -> None:
        from reactor_twin.reactors.batch import BatchReactor

        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            BatchReactor(name="bad", num_species=2, params={"T": 350.0})

    def test_semi_batch_missing_param_raises_configuration_error(self) -> None:
        from reactor_twin.reactors.semi_batch import SemiBatchReactor

        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            SemiBatchReactor(
                name="bad",
                num_species=2,
                params={"V": 50.0, "T": 350.0, "C_in": [2.0, 0.0]},
            )

    def test_pfr_missing_param_raises_configuration_error(self) -> None:
        from reactor_twin.reactors.pfr import PlugFlowReactor

        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            PlugFlowReactor(
                name="bad",
                num_species=2,
                params={"u": 0.1, "D": 0.01, "C_in": [1.0, 0.0], "T": 350.0},
            )

    def test_ode_func_unknown_activation_raises_validation_error(self) -> None:
        from reactor_twin.core.ode_func import MLPODEFunc

        with pytest.raises(ValidationError, match="Unknown activation"):
            MLPODEFunc(state_dim=3, activation="invalid_act")

    def test_constraint_invalid_mode_raises_validation_error(self) -> None:
        from reactor_twin.physics.positivity import PositivityConstraint

        with pytest.raises(ValidationError, match="mode must be"):
            PositivityConstraint(mode="invalid_mode")
