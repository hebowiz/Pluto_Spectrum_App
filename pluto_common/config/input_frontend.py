"""Common Pluto input-plane amplitude settings for SA and VSA."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PLUTO_CALIBRATION_OFFSET_DB = -62.0
DEFAULT_INTERNAL_GAIN_DB = 30.0
DEFAULT_EXTERNAL_ATTENUATION_DB = 30.0
DEFAULT_EXTERNAL_GAIN_DB = 0.0


@dataclass(frozen=True)
class InputPowerCorrection:
    """One shared definition of Pluto and external-path gain correction."""

    calibration_offset_db: float = DEFAULT_PLUTO_CALIBRATION_OFFSET_DB
    internal_gain_db: float = DEFAULT_INTERNAL_GAIN_DB
    external_attenuation_db: float = DEFAULT_EXTERNAL_ATTENUATION_DB
    external_gain_db: float = DEFAULT_EXTERNAL_GAIN_DB
    frequency_dependent_offset_db: float = 0.0

    @property
    def input_correction_db(self) -> float:
        """Correction from Pluto ADC plane to the selected external plane."""

        return float(
            self.external_attenuation_db
            - self.internal_gain_db
            - self.external_gain_db
        )

    @property
    def total_power_offset_db(self) -> float:
        return float(
            self.calibration_offset_db
            + self.frequency_dependent_offset_db
            + self.input_correction_db
        )
