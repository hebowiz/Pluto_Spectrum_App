import numpy as np
import pytest

from pluto_sa.vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    BLUETOOTH_HDT_MAPPING,
    GRAY_MAPPING,
    logical_to_phase_indices,
    phase_indices_to_logical_symbols,
    psk_constellation,
    reverse_symbol_bits,
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


def test_symbol_bit_reversal_matches_rs_lsb_display_numbering():
    np.testing.assert_array_equal(
        reverse_symbol_bits(np.arange(4), 4), [0, 2, 1, 3]
    )
    np.testing.assert_array_equal(
        reverse_symbol_bits(np.arange(8), 8), [0, 4, 2, 6, 1, 5, 3, 7]
    )
    values = np.asarray([0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(
        reverse_symbol_bits(reverse_symbol_bits(values, 8), 8), values
    )


def test_bluetooth_hdt_16qam_constellation_uses_spec_scale():
    constellation = psk_constellation(
        ModulationKind.QAM16, BLUETOOTH_HDT_MAPPING
    )

    assert constellation.shape == (16,)
    assert np.unique(constellation.real).size == 4
    assert np.unique(constellation.imag).size == 4
    assert np.mean(np.abs(constellation) ** 2) == pytest.approx(0.1)
