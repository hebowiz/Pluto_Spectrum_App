import numpy as np
import pytest

from pluto_sa.vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    GRAY_MAPPING,
    logical_to_phase_indices,
    phase_indices_to_logical_symbols,
)
from pluto_sa.vsa.model import ModulationKind, SignalDescription


def test_rs_gray_phase_indices_match_generic_dpsk_tables():
    np.testing.assert_array_equal(
        logical_to_phase_indices(ModulationKind.PI4_DQPSK, GRAY_MAPPING),
        [0, 1, 3, 2],
    )
    np.testing.assert_array_equal(
        logical_to_phase_indices(ModulationKind.DPSK8, GRAY_MAPPING),
        [0, 1, 3, 2, 6, 7, 5, 4],
    )


def test_bluetooth_edr_mapping_matches_core_phase_tables():
    np.testing.assert_array_equal(
        logical_to_phase_indices(ModulationKind.PI4_DQPSK, BLUETOOTH_EDR_MAPPING),
        [0, 1, 3, 2],
    )
    np.testing.assert_array_equal(
        logical_to_phase_indices(ModulationKind.DPSK8, BLUETOOTH_EDR_MAPPING),
        [0, 1, 3, 2, 7, 6, 4, 5],
    )
    np.testing.assert_array_equal(
        phase_indices_to_logical_symbols(
            ModulationKind.DPSK8,
            BLUETOOTH_EDR_MAPPING,
            np.arange(8),
        ),
        [0, 1, 3, 2, 6, 7, 5, 4],
    )


def test_bluetooth_mapping_is_rejected_for_non_edr_modulation():
    with pytest.raises(ValueError, match="requires pi/4-DQPSK or 8DPSK"):
        SignalDescription(
            modulation=ModulationKind.QPSK,
            symbol_rate_hz=1_000_000.0,
            symbol_mapping=BLUETOOTH_EDR_MAPPING,
        )
