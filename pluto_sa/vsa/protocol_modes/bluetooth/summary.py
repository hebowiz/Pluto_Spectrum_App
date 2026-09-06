"""Stable Result Summary view models for Bluetooth Dedicated VSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .rf_measurement.limits import (
    BR_CARRIER_DRIFT_LIMIT_HZ_BY_SLOTS,
    BR_CARRIER_DRIFT_RATE_LIMIT_HZ_PER_50_US,
    BR_DELTA_F1_AVG_RANGE_HZ,
    BR_DELTA_F2_P999_MIN_HZ,
    BR_INITIAL_CARRIER_FREQUENCY_LIMIT_HZ,
    EDR_COMBINED_FREQUENCY_ERROR_RANGE_HZ,
    EDR_DEVM_LIMITS,
    EDR_GUARD_TIME_RANGE_S,
    EDR_INITIAL_FREQUENCY_ERROR_RANGE_HZ,
    EDR_RELATIVE_TRANSMIT_POWER_RANGE_DB,
    EDR_REQUIRED_DIFFERENTIAL_PHASE_PACKETS,
    EDR_REQUIRED_DIFFERENTIAL_PHASE_VALID_PACKETS,
    EDR_REQUIRED_DEVM_BLOCKS,
    EDR_REQUIRED_GUARD_PACKETS,
    EDR_REQUIRED_GUARD_VALID_PACKETS,
    EDR_REQUIRED_SYNC_PACKETS,
    EDR_REQUIRED_TRAILER_PACKETS,
    EDR_RESIDUAL_FREQUENCY_ERROR_RANGE_HZ,
    EDR_MAX_TRAILER_BIT_ERRORS,
    FSK_DELTA_F2_RATIO_MIN,
    LE_CARRIER_DRIFT_LIMIT_HZ,
    LE_CARRIER_DRIFT_RATE_LIMIT_HZ_PER_50_US,
    LE_CARRIER_FREQUENCY_LIMIT_HZ,
    LE_DELTA_F1_AVG_RANGE_HZ,
    LE_DELTA_F2_P999_MIN_HZ,
    OUTPUT_POWER_LIMIT_DEPENDENCY,
)
from .rf_measurement.model import BluetoothRFMeasurementResult

if TYPE_CHECKING:
    from .model import BluetoothDedicatedResult, BluetoothMetric


RF_PHY_MEASUREMENTS = "RF PHY Measurements"
REFERENCE_INFORMATION = "Reference Information"
EM_DASH = "\N{EM DASH}"


@dataclass(frozen=True)
class BluetoothSummaryRow:
    section: str
    test_item: str
    value: str
    limit: str = EM_DASH
    result: str = EM_DASH
    metric_id: str = ""


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _display(value: object, unit: str, scale: float = 1.0) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number / scale:+.3f} {unit}"


def _display_unsigned(value: object, unit: str, scale: float = 1.0) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number / scale:.3f} {unit}"


def _display_percent_fraction(value: object) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{100.0 * number:.2f} %"


def _measurement(
    result: "BluetoothDedicatedResult", test_case_id: str
) -> BluetoothRFMeasurementResult | None:
    return next(
        (
            item
            for item in result.metadata.get("rf_measurements", ())
            if isinstance(item, BluetoothRFMeasurementResult)
            and item.test_case_id == test_case_id
        ),
        None,
    )


def _metric_map(
    result: "BluetoothDedicatedResult",
) -> Mapping[str, "BluetoothMetric"]:
    return {metric.metric_id: metric for metric in result.metrics}


def _eligibility_value(measurement: BluetoothRFMeasurementResult | None) -> str:
    if measurement is None:
        return "N/A"
    if measurement.eligibility.eligible:
        return "Eligible"
    reasons = "; ".join(measurement.eligibility.reasons)
    return "N/A" if not reasons else f"N/A - {reasons}"


def _pending_result(
    result: "BluetoothDedicatedResult",
    measurement: BluetoothRFMeasurementResult | None,
    *,
    complete: bool,
) -> str:
    if str(result.profile) != "rf_phy_test":
        return "N/A"
    if measurement is None:
        return "N/A"
    non_progress_reasons = tuple(
        reason
        for reason in measurement.eligibility.reasons
        if not reason.startswith("requires ")
    )
    if non_progress_reasons:
        return "N/A"
    return "PASS" if complete else "MEASURING"


def build_hdt_summary(result: "BluetoothDedicatedResult") -> tuple[BluetoothSummaryRow, ...]:
    metrics = _metric_map(result)
    measurement_ids = (
        "sig_hdt_output_power",
        "sig_hdt_header_evm_rms",
        "sig_hdt_payload_evm_rms",
        "sig_hdt_center_frequency_deviation",
        "sig_hdt_frequency_offset_change",
        "sig_hdt_symbol_timing_accuracy",
        "sig_hdt_pre_packet_emissions",
    )
    reference_ids = (
        "detected_phy",
        "sig_eligibility",
        "sig_hdt_preamble_carrier_error",
        "sig_hdt_payload_carrier_error",
        "sig_hdt_header_average_power",
        "sig_hdt_payload_average_power",
        "sig_hdt_relative_power",
        "sig_hdt_training_correlation",
        "sig_hdt_evm_packets_evaluated",
    )
    rows: list[BluetoothSummaryRow] = []
    for section, identifiers in (
        (RF_PHY_MEASUREMENTS, measurement_ids),
        (REFERENCE_INFORMATION, reference_ids),
    ):
        for identifier in identifiers:
            metric = metrics[identifier]
            rows.append(
                BluetoothSummaryRow(
                    section,
                    metric.label,
                    metric.display,
                    metric.limit if section == RF_PHY_MEASUREMENTS else EM_DASH,
                    metric.result if section == RF_PHY_MEASUREMENTS else EM_DASH,
                    identifier,
                )
            )
    return tuple(rows)


def _packet_slots(packet_type: str | None) -> int:
    text = str(packet_type or "")
    if text.endswith("5"):
        return 5
    if text.endswith("3"):
        return 3
    return 1


def build_fsk_summary(
    result: "BluetoothDedicatedResult",
) -> tuple[BluetoothSummaryRow, ...]:
    phy = result.packet.phy_name
    is_le = phy.startswith("LE ")
    measurement = _measurement(result, "bluetooth.fsk")
    aggregate = result.metadata.get("rf_capture_aggregate")
    if not isinstance(aggregate, BluetoothRFMeasurementResult) or not aggregate.test_case_id.endswith(
        "fsk.aggregate"
    ):
        aggregate = None
    per_metrics = measurement.metrics if measurement is not None else {}
    aggregate_metrics = aggregate.metrics if aggregate is not None else {}
    f1 = aggregate_metrics.get("delta_f1_avg_hz", per_metrics.get("delta_f1_avg_hz"))
    per_f2 = (
        np.asarray(measurement.arrays.get("delta_f2_max_hz", ()), dtype=np.float64)
        if measurement is not None
        else np.empty(0, dtype=np.float64)
    )
    per_f2 = per_f2[np.isfinite(per_f2)]
    per_f2_floor = (
        float(np.percentile(per_f2, 0.1)) if per_f2.size else None
    )
    f2_floor = aggregate_metrics.get("delta_f2_p999_floor_hz", per_f2_floor)
    ratio = aggregate_metrics.get("delta_f2_ratio")
    f1_count = int(
        aggregate_metrics.get(
            "delta_f1_packet_count", 1 if _number(per_metrics.get("delta_f1_avg_hz")) is not None else 0
        )
    )
    f2_count = int(
        aggregate_metrics.get("delta_f2_packet_count", 1 if per_f2.size else 0)
    )
    aggregate_complete = f1_count > 0 and f2_count > 0
    qualification = aggregate if aggregate is not None else measurement
    aggregate_state = _pending_result(
        result, qualification, complete=aggregate_complete
    )

    if is_le:
        f1_low, f1_high = LE_DELTA_F1_AVG_RANGE_HZ[phy]
        carrier_label = "Carrier frequency offset"
        carrier_limit = "\N{PLUS-MINUS SIGN}150 kHz"
        carrier_limit_hz = LE_CARRIER_FREQUENCY_LIMIT_HZ
        drift_limit = "< 50 kHz"
        drift_limit_hz = LE_CARRIER_DRIFT_LIMIT_HZ
        drift_rate_limit_hz = LE_CARRIER_DRIFT_RATE_LIMIT_HZ_PER_50_US
    else:
        f1_low, f1_high = BR_DELTA_F1_AVG_RANGE_HZ
        carrier_label = "Initial carrier frequency"
        carrier_limit = "\N{PLUS-MINUS SIGN}75 kHz"
        carrier_limit_hz = BR_INITIAL_CARRIER_FREQUENCY_LIMIT_HZ
        slot_limit = BR_CARRIER_DRIFT_LIMIT_HZ_BY_SLOTS[
            _packet_slots(result.packet.packet_type)
        ]
        drift_limit = f"\N{PLUS-MINUS SIGN}{slot_limit / 1e3:.0f} kHz"
        drift_limit_hz = slot_limit
        drift_rate_limit_hz = BR_CARRIER_DRIFT_RATE_LIMIT_HZ_PER_50_US

    def aggregate_verdict(passed: bool) -> str:
        if aggregate_state != "PASS":
            return aggregate_state
        return "PASS" if passed else "FAIL"

    def modulation_verdict(value: float | None, passed: bool) -> str:
        # The prescribed 11110000/10101010 payload patterns qualify only the
        # corresponding modulation-characteristic rows.  An arbitrary payload
        # must not turn otherwise valid carrier measurements into N/A.
        if value is None:
            return aggregate_state if f1_count > 0 or f2_count > 0 else "N/A"
        return aggregate_verdict(passed)

    f1_number = _number(f1)
    f2_number = _number(f2_floor)
    ratio_number = _number(ratio)
    carrier = _number(per_metrics.get("initial_carrier_error_hz"))
    carrier_result = (
        "N/A"
        if carrier is None or measurement is None or not measurement.eligibility.eligible
        else "PASS"
        if abs(carrier) <= carrier_limit_hz
        else "FAIL"
    )
    drift = _number(per_metrics.get("max_drift_from_f0_hz"))
    drift_result = (
        "N/A"
        if drift is None or measurement is None or not measurement.eligibility.eligible
        else "PASS"
        if abs(drift) <= drift_limit_hz
        else "FAIL"
    )
    drift_rate = _number(per_metrics.get("max_drift_rate_hz"))
    drift_rate_result = (
        "N/A"
        if drift_rate is None
        or measurement is None
        or not measurement.eligibility.eligible
        else "PASS"
        if abs(drift_rate) <= drift_rate_limit_hz
        else "FAIL"
    )
    rows = (
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "Output power",
            _display(per_metrics.get("pavg_dbm"), "dBm"),
            OUTPUT_POWER_LIMIT_DEPENDENCY,
            "N/A",
            "output_power",
        ),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "\N{GREEK CAPITAL LETTER DELTA}f1avg",
            _display_unsigned(f1, "kHz", 1e3),
            f"{f1_low / 1e3:.0f} kHz \N{LESS-THAN OR EQUAL TO} \N{GREEK CAPITAL LETTER DELTA}f1avg \N{LESS-THAN OR EQUAL TO} {f1_high / 1e3:.0f} kHz",
            modulation_verdict(
                f1_number,
                f1_number is not None and f1_low <= f1_number <= f1_high
            ),
            "delta_f1_avg",
        ),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "99.9% \N{GREEK CAPITAL LETTER DELTA}f2max",
            _display_unsigned(f2_floor, "kHz", 1e3),
            (
                f"\N{GREATER-THAN OR EQUAL TO} {BR_DELTA_F2_P999_MIN_HZ / 1e3:.0f} kHz"
                if not is_le
                else f"\N{GREATER-THAN OR EQUAL TO} {LE_DELTA_F2_P999_MIN_HZ[phy] / 1e3:.0f} kHz"
            ),
            (
                modulation_verdict(
                    f2_number,
                    f2_number is not None
                    and f2_number >= BR_DELTA_F2_P999_MIN_HZ
                )
                if not is_le
                else modulation_verdict(
                    f2_number,
                    f2_number is not None
                    and f2_number >= LE_DELTA_F2_P999_MIN_HZ[phy]
                )
            ),
            "delta_f2_p999",
        ),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "\N{GREEK CAPITAL LETTER DELTA}f2avg / \N{GREEK CAPITAL LETTER DELTA}f1avg",
            "N/A" if ratio_number is None else f"{ratio_number:.3f}",
            f"\N{GREATER-THAN OR EQUAL TO} {FSK_DELTA_F2_RATIO_MIN:.1f}",
            modulation_verdict(
                ratio_number,
                ratio_number is not None and ratio_number >= FSK_DELTA_F2_RATIO_MIN
            ),
            "delta_f2_ratio",
        ),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            carrier_label,
            _display(per_metrics.get("initial_carrier_error_hz"), "kHz", 1e3),
            carrier_limit,
            carrier_result,
            "initial_carrier_frequency",
        ),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "Carrier frequency drift",
            _display(per_metrics.get("max_drift_from_f0_hz"), "kHz", 1e3),
            drift_limit,
            drift_result,
            "carrier_frequency_drift",
        ),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "Carrier frequency drift rate",
            _display(per_metrics.get("max_drift_rate_hz"), "kHz", 1e3),
            "\N{LESS-THAN OR EQUAL TO} 20 kHz / 50 \N{MICRO SIGN}s",
            drift_rate_result,
            "carrier_frequency_drift_rate",
        ),
    )
    old = _metric_map(result)
    pattern = None if measurement is None else measurement.metadata.get("payload_pattern")
    packet_count = int(aggregate_metrics.get("packet_count", 1 if measurement else 0))
    reference: list[BluetoothSummaryRow] = [
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Detected PHY", phy, metric_id="detected_phy"),
        BluetoothSummaryRow(
            REFERENCE_INFORMATION,
            "RF Test Eligibility",
            _eligibility_value(measurement),
            metric_id="rf_test_eligibility",
        ),
        BluetoothSummaryRow(
            REFERENCE_INFORMATION,
            "Packet Type" if not is_le else "Payload Pattern",
            str(result.packet.packet_type or "N/A") if not is_le else str(pattern or "N/A"),
            metric_id="packet_type" if not is_le else "payload_pattern",
        ),
    ]
    if is_le:
        reference.append(
            BluetoothSummaryRow(
                REFERENCE_INFORMATION,
                "Sync Correlation",
                old.get("correlation").display if old.get("correlation") else "N/A",
                metric_id="sync_correlation",
            )
        )
    else:
        reference.extend(
            (
                BluetoothSummaryRow(
                    REFERENCE_INFORMATION,
                    "Payload Pattern",
                    str(pattern or "N/A"),
                    metric_id="payload_pattern",
                ),
                BluetoothSummaryRow(
                    REFERENCE_INFORMATION,
                    "Access Code Correlation",
                    old.get("correlation").display if old.get("correlation") else "N/A",
                    metric_id="access_code_correlation",
                ),
            )
        )
    reference.extend(
        (
            BluetoothSummaryRow(
                REFERENCE_INFORMATION,
                "Packets Evaluated",
                str(packet_count),
                metric_id="packets_evaluated",
            ),
            BluetoothSummaryRow(
                REFERENCE_INFORMATION,
                "Peak Power",
                _display(per_metrics.get("ppk_dbm"), "dBm"),
                metric_id="peak_power",
            ),
            BluetoothSummaryRow(
                REFERENCE_INFORMATION,
                "Mean abs. FSK deviation",
                _display_unsigned(
                    per_metrics.get("mean_abs_fsk_deviation_hz"), "kHz", 1e3
                ),
                metric_id="mean_abs_fsk_deviation",
            ),
            BluetoothSummaryRow(
                REFERENCE_INFORMATION,
                "99.9% FSK deviation",
                _display_unsigned(
                    per_metrics.get("p999_abs_fsk_deviation_hz"), "kHz", 1e3
                ),
                metric_id="p999_fsk_deviation",
            ),
            BluetoothSummaryRow(
                REFERENCE_INFORMATION,
                "Max. FSK deviation",
                _display_unsigned(
                    per_metrics.get("max_abs_fsk_deviation_hz"), "kHz", 1e3
                ),
                metric_id="max_fsk_deviation",
            ),
        )
    )
    return (*rows, *reference)


def build_edr_summary(
    result: "BluetoothDedicatedResult",
) -> tuple[BluetoothSummaryRow, ...]:
    phy = result.packet.phy_name
    measurement = _measurement(result, "bluetooth.edr")
    metrics = measurement.metrics if measurement is not None else {}
    aggregate = result.metadata.get("rf_capture_aggregate")
    if not isinstance(aggregate, BluetoothRFMeasurementResult) or not aggregate.test_case_id.endswith(
        "edr.aggregate"
    ):
        aggregate = None
    aggregate_metrics = aggregate.metrics if aggregate is not None else {}
    block_count = int(aggregate_metrics.get("block_count", metrics.get("block_count", 0)))
    devm_values = {
        "rms": aggregate_metrics.get("rms_devm_worst", metrics.get("rms_devm_worst")),
        "p99": aggregate_metrics.get("devm_99_percentile", metrics.get("devm_99_percentile")),
        "peak": aggregate_metrics.get("peak_devm_worst", metrics.get("peak_devm_worst")),
    }
    limits = EDR_DEVM_LIMITS[phy]
    measuring = block_count < EDR_REQUIRED_DEVM_BLOCKS
    eligible = (
        str(result.profile) == "rf_phy_test"
        and measurement is not None
        and measurement.eligibility.eligible
    )

    def worst_array(name: str) -> float | None:
        if measurement is None:
            return None
        values = np.asarray(measurement.arrays.get(name, ()), dtype=np.float64)
        values = values[np.isfinite(values)]
        return None if not values.size else float(values[np.argmax(np.abs(values))])

    initial = aggregate_metrics.get(
        "initial_frequency_error_worst_hz",
        metrics.get("initial_frequency_error_hz"),
    )
    omega0 = aggregate_metrics.get(
        "omega0_worst_hz", metrics.get("omega0_worst_hz")
    )
    combined = aggregate_metrics.get(
        "combined_frequency_error_worst_hz",
        worst_array("omega_i_plus_omega0_hz"),
    )

    def range_result(value: object, bounds: tuple[float, float]) -> str:
        number = _number(value)
        if not eligible or number is None:
            return "N/A"
        return "PASS" if bounds[0] < number < bounds[1] else "FAIL"

    def devm_result(key: str) -> str:
        value = _number(devm_values[key])
        if not eligible:
            return "N/A"
        if key in {"rms", "peak"} and value is not None and value > limits[key]:
            return "FAIL"
        if measuring:
            return "MEASURING"
        return "PASS" if value is not None and value <= limits[key] else "FAIL"

    relative = _number(metrics.get("relative_power_db"))
    low, high = EDR_RELATIVE_TRANSMIT_POWER_RANGE_DB
    relative_result = (
        "N/A"
        if relative is None or not eligible
        else "PASS"
        if low < relative < high
        else "FAIL"
    )
    guard_value = aggregate_metrics.get(
        "guard_time_mean_s", metrics.get("guard_time_s")
    )
    guard_count = int(
        aggregate_metrics.get(
            "guard_time_packet_count", 1 if _number(metrics.get("guard_time_s")) is not None else 0
        )
    )
    guard_valid = int(
        aggregate_metrics.get(
            "guard_time_valid_packet_count",
            int(
                _number(metrics.get("guard_time_s")) is not None
                and EDR_GUARD_TIME_RANGE_S[0]
                <= float(metrics["guard_time_s"])
                <= EDR_GUARD_TIME_RANGE_S[1]
            ),
        )
    )
    phase_count = int(
        aggregate_metrics.get(
            "differential_phase_packet_count",
            1 if metrics.get("payload_bit_errors") is not None else 0,
        )
    )
    phase_valid = int(
        aggregate_metrics.get(
            "differential_phase_valid_packet_count",
            int(metrics.get("payload_bit_errors") == 0),
        )
    )
    phase_errors = aggregate_metrics.get(
        "payload_bit_errors", metrics.get("payload_bit_errors")
    )
    sync_count = int(
        aggregate_metrics.get(
            "sync_packet_count", 1 if metrics.get("sync_bit_errors") is not None else 0
        )
    )
    sync_errors = int(
        aggregate_metrics.get("sync_symbol_errors", metrics.get("sync_bit_errors", 0))
    )
    trailer_count = int(
        aggregate_metrics.get(
            "trailer_packet_count",
            1 if metrics.get("trailer_bit_errors") is not None else 0,
        )
    )
    trailer_errors = aggregate_metrics.get(
        "trailer_symbol_errors", metrics.get("trailer_bit_errors")
    )

    def quota_result(count: int, valid: int, required: int, required_valid: int) -> str:
        if not eligible:
            return "N/A"
        if valid + max(0, required - count) < required_valid:
            return "FAIL"
        if count < required:
            return "MEASURING"
        return "PASS" if valid >= required_valid else "FAIL"

    guard_result = quota_result(
        guard_count,
        guard_valid,
        EDR_REQUIRED_GUARD_PACKETS,
        EDR_REQUIRED_GUARD_VALID_PACKETS,
    )
    phase_result = quota_result(
        phase_count,
        phase_valid,
        EDR_REQUIRED_DIFFERENTIAL_PHASE_PACKETS,
        EDR_REQUIRED_DIFFERENTIAL_PHASE_VALID_PACKETS,
    )
    sync_result = (
        "N/A"
        if not eligible
        else "FAIL"
        if sync_errors > 0
        else "MEASURING"
        if sync_count < EDR_REQUIRED_SYNC_PACKETS
        else "PASS"
    )
    rows = (
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "Output power",
            _display(metrics.get("output_power_dbm"), "dBm"),
            OUTPUT_POWER_LIMIT_DEPENDENCY,
            "N/A",
            "output_power",
        ),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Relative transmit power", _display(metrics.get("relative_power_db"), "dB"), "-4 dB < value < +1 dB", relative_result, "relative_transmit_power"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Initial frequency error \N{GREEK SMALL LETTER OMEGA}i", _display(initial, "kHz", 1e3), "-75 kHz < \N{GREEK SMALL LETTER OMEGA}i < +75 kHz", range_result(initial, EDR_INITIAL_FREQUENCY_ERROR_RANGE_HZ), "omega_i"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Residual frequency error \N{GREEK SMALL LETTER OMEGA}0", _display(omega0, "kHz", 1e3), "-10 kHz < \N{GREEK SMALL LETTER OMEGA}0 < +10 kHz", range_result(omega0, EDR_RESIDUAL_FREQUENCY_ERROR_RANGE_HZ), "omega_0"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "\N{GREEK SMALL LETTER OMEGA}i + \N{GREEK SMALL LETTER OMEGA}0", _display(combined, "kHz", 1e3), "-75 kHz < value < +75 kHz", range_result(combined, EDR_COMBINED_FREQUENCY_ERROR_RANGE_HZ), "omega_i_plus_omega_0"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "RMS DEVM", _display_percent_fraction(devm_values["rms"]), f"\N{LESS-THAN OR EQUAL TO} {100 * limits['rms']:.0f} %", devm_result("rms"), "rms_devm"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "99% DEVM", _display_percent_fraction(devm_values["p99"]), f"\N{LESS-THAN OR EQUAL TO} {100 * limits['p99']:.0f} %", devm_result("p99"), "p99_devm"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Peak DEVM", _display_percent_fraction(devm_values["peak"]), f"\N{LESS-THAN OR EQUAL TO} {100 * limits['peak']:.0f} %", devm_result("peak"), "peak_devm"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Guard time", _display_unsigned(guard_value, "\N{MICRO SIGN}s", 1e-6), "4.60\N{EN DASH}5.40 \N{MICRO SIGN}s", guard_result, "guard_time"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Differential phase encoding", "N/A" if phase_errors is None else f"{int(phase_errors)} bit error(s)", "\N{GREATER-THAN OR EQUAL TO} 99 / 100 packets with 0 bit errors", phase_result, "differential_phase_encoding"),
        BluetoothSummaryRow(RF_PHY_MEASUREMENTS, "Synchronization sequence", f"{sync_errors} bit error(s)", "0 bit errors / 50 packets", sync_result, "synchronization_sequence"),
        BluetoothSummaryRow(
            RF_PHY_MEASUREMENTS,
            "Trailer",
            "N/A" if trailer_errors is None else f"{int(trailer_errors)} bit error(s)",
            "\N{LESS-THAN OR EQUAL TO} 1 bit error / 50 packets",
            (
                "N/A"
                if not eligible or trailer_errors is None
                else "FAIL"
                if int(trailer_errors) > EDR_MAX_TRAILER_BIT_ERRORS
                else "MEASURING"
                if trailer_count < EDR_REQUIRED_TRAILER_PACKETS
                else "PASS"
            ),
            "trailer",
        ),
    )
    old = _metric_map(result)
    reference = (
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Detected PHY", phy, metric_id="detected_phy"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "RF Test Eligibility", _eligibility_value(measurement), metric_id="rf_test_eligibility"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Packet Type", str(result.packet.packet_type or "N/A"), metric_id="packet_type"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Payload Pattern", str(measurement.metadata.get("payload_pattern", "N/A") if measurement else "N/A"), metric_id="payload_pattern"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Access Code Correlation", old.get("correlation").display if old.get("correlation") else "N/A", metric_id="access_code_correlation"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "PGFSK", _display(metrics.get("pgfsk_dbm"), "dBm"), metric_id="pgfsk"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "PDPSK", _display(metrics.get("pdpsk_dbm"), "dBm"), metric_id="pdpsk"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "DEVM Blocks Evaluated", f"{block_count} / {EDR_REQUIRED_DEVM_BLOCKS}", metric_id="devm_blocks_evaluated"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Guard Time Packets Evaluated", f"{guard_count} / {EDR_REQUIRED_GUARD_PACKETS}", metric_id="guard_time_packets_evaluated"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Guard Time Valid Packets", f"{guard_valid} / {EDR_REQUIRED_GUARD_PACKETS}", metric_id="guard_time_valid_packets"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Differential Phase Packets Evaluated", f"{phase_count} / {EDR_REQUIRED_DIFFERENTIAL_PHASE_PACKETS}", metric_id="differential_phase_packets_evaluated"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Differential Phase Valid Packets", f"{phase_valid} / {EDR_REQUIRED_DIFFERENTIAL_PHASE_PACKETS}", metric_id="differential_phase_valid_packets"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Sync Packets Evaluated", f"{sync_count} / {EDR_REQUIRED_SYNC_PACKETS}", metric_id="sync_packets_evaluated"),
        BluetoothSummaryRow(REFERENCE_INFORMATION, "Trailer Packets Evaluated", f"{trailer_count} / {EDR_REQUIRED_TRAILER_PACKETS}", metric_id="trailer_packets_evaluated"),
    )
    return (*rows, *reference)


def build_bluetooth_summary(
    result: "BluetoothDedicatedResult",
) -> tuple[BluetoothSummaryRow, ...]:
    phy = result.packet.phy_name
    if phy.startswith("HDT"):
        return build_hdt_summary(result)
    if phy.startswith("EDR"):
        return build_edr_summary(result)
    return build_fsk_summary(result)
