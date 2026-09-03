"""Multi-packet aggregation for Bluetooth RF test measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (
    BluetoothRFMeasurementResult,
    RFTestEligibility,
    RFTestVerdict,
)
from .limits import (
    BR_DELTA_F1_AVG_RANGE_HZ,
    BR_DELTA_F2_P999_MIN_HZ,
    EDR_DEVM_LIMITS,
    EDR_GUARD_TIME_RANGE_S,
    EDR_REQUIRED_DIFFERENTIAL_PHASE_PACKETS,
    EDR_REQUIRED_DEVM_BLOCKS,
    EDR_REQUIRED_GUARD_PACKETS,
    EDR_REQUIRED_SYNC_PACKETS,
    EDR_REQUIRED_TRAILER_PACKETS,
    FSK_DELTA_F2_RATIO_MIN,
)


@dataclass
class BluetoothRFTestAccumulator:
    """Collect evidence without converting incomplete tests into a verdict."""

    _results: list[BluetoothRFMeasurementResult] = field(default_factory=list)

    def add(self, result: BluetoothRFMeasurementResult) -> None:
        self._results.append(result)

    @property
    def results(self) -> tuple[BluetoothRFMeasurementResult, ...]:
        return tuple(self._results)

    def aggregate_edr(
        self, *, required_blocks: int = EDR_REQUIRED_DEVM_BLOCKS
    ) -> BluetoothRFMeasurementResult:
        edr = [result for result in self._results if result.test_case_id == "bluetooth.edr"]
        base_reasons = tuple(dict.fromkeys(
            reason
            for result in edr
            for reason in result.eligibility.reasons
        ))
        rms = np.concatenate(
            [result.arrays.get("block_rms_devm", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        peak = np.concatenate(
            [result.arrays.get("block_peak_devm", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        symbol_devm = np.concatenate(
            [result.arrays.get("symbol_devm", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        omega0 = np.concatenate(
            [result.arrays.get("omega0_hz", np.empty(0)) for result in edr]
        ) if edr else np.empty(0)
        combined_frequency = np.concatenate(
            [
                result.arrays.get("omega_i_plus_omega0_hz", np.empty(0))
                for result in edr
            ]
        ) if edr else np.empty(0)
        omega_i = np.asarray(
            [
                float(result.metrics["initial_frequency_error_hz"])
                for result in edr
                if result.metrics.get("initial_frequency_error_hz") is not None
            ],
            dtype=np.float64,
        )
        guard_time = np.asarray(
            [
                float(result.metrics["guard_time_s"])
                for result in edr
                if result.metrics.get("guard_time_s") is not None
            ],
            dtype=np.float64,
        )
        payload_bit_errors = np.asarray(
            [
                int(result.metrics["payload_bit_errors"])
                for result in edr
                if result.metrics.get("payload_bit_errors") is not None
            ],
            dtype=np.int64,
        )
        sync_bit_errors = np.asarray(
            [
                int(
                    result.metrics.get(
                        "sync_bit_errors", result.metrics["sync_symbol_errors"]
                    )
                )
                for result in edr
                if result.metrics.get("sync_symbol_errors") is not None
            ],
            dtype=np.int64,
        )
        trailer_bit_errors = np.asarray(
            [
                int(
                    result.metrics.get(
                        "trailer_bit_errors",
                        result.metrics["trailer_symbol_errors"],
                    )
                )
                for result in edr
                if result.metrics.get("trailer_symbol_errors") is not None
            ],
            dtype=np.int64,
        )
        guard_evaluated = guard_time[:EDR_REQUIRED_GUARD_PACKETS]
        phase_evaluated = payload_bit_errors[
            :EDR_REQUIRED_DIFFERENTIAL_PHASE_PACKETS
        ]
        sync_evaluated = sync_bit_errors[:EDR_REQUIRED_SYNC_PACKETS]
        trailer_evaluated = trailer_bit_errors[:EDR_REQUIRED_TRAILER_PACKETS]
        rms = rms[: int(required_blocks)]
        peak = peak[: int(required_blocks)]
        symbol_devm = symbol_devm[: int(required_blocks) * 50]
        omega0 = omega0[: int(required_blocks)]
        combined_frequency = combined_frequency[: int(required_blocks)]
        block_count = int(rms.size)
        reasons = base_reasons
        if block_count < int(required_blocks):
            reasons = (*reasons, f"requires {required_blocks} DEVM blocks; {block_count} available")
        eligibility = RFTestEligibility.from_reasons(reasons)
        modulation = next(
            (
                str(result.metadata.get("modulation"))
                for result in edr
                if result.metadata.get("modulation")
            ),
            "",
        )
        if "8" in modulation:
            limits = EDR_DEVM_LIMITS["EDR 3M"]
        else:
            limits = EDR_DEVM_LIMITS["EDR 2M"]
        rms_limit = limits["rms"]
        percentile_limit = limits["p99"]
        peak_limit = limits["peak"]
        rms_worst = float(np.max(rms)) if rms.size else None
        peak_worst = float(np.max(peak)) if peak.size else None
        percentile_99 = (
            float(np.percentile(symbol_devm, 99.0)) if symbol_devm.size else None
        )
        verdict = RFTestVerdict.NOT_APPLICABLE
        hard_devm_failure = (
            rms_worst is not None
            and rms_worst > rms_limit
            or peak_worst is not None
            and peak_worst > peak_limit
        )
        if not base_reasons and hard_devm_failure:
            verdict = RFTestVerdict.FAIL
            eligibility = RFTestEligibility(True)
        elif (
            not base_reasons
            and block_count >= int(required_blocks)
            and rms_worst is not None
            and percentile_99 is not None
            and peak_worst is not None
        ):
            verdict = (
                RFTestVerdict.PASS
                if rms_worst <= rms_limit
                and percentile_99 <= percentile_limit
                and peak_worst <= peak_limit
                else RFTestVerdict.FAIL
            )
        return BluetoothRFMeasurementResult(
            "bluetooth.edr.aggregate",
            eligibility,
            verdict,
            metrics={
                "packet_count": len(edr),
                "block_count": block_count,
                "rms_devm_worst": rms_worst,
                "devm_99_percentile": percentile_99,
                "peak_devm_worst": peak_worst,
                "initial_frequency_error_worst_hz": (
                    None if not omega_i.size else float(omega_i[np.argmax(np.abs(omega_i))])
                ),
                "omega0_worst_hz": (
                    None if not omega0.size else float(omega0[np.argmax(np.abs(omega0))])
                ),
                "combined_frequency_error_worst_hz": (
                    None
                    if not combined_frequency.size
                    else float(combined_frequency[np.argmax(np.abs(combined_frequency))])
                ),
                "guard_time_mean_s": (
                    None
                    if not guard_evaluated.size
                    else float(np.mean(guard_evaluated))
                ),
                "guard_time_packet_count": int(guard_evaluated.size),
                "guard_time_valid_packet_count": int(
                    np.count_nonzero(
                        (guard_evaluated >= EDR_GUARD_TIME_RANGE_S[0])
                        & (guard_evaluated <= EDR_GUARD_TIME_RANGE_S[1])
                    )
                ),
                "differential_phase_packet_count": int(phase_evaluated.size),
                "differential_phase_valid_packet_count": int(
                    np.count_nonzero(phase_evaluated == 0)
                ),
                "payload_bit_errors": int(np.sum(phase_evaluated)),
                "sync_packet_count": int(sync_evaluated.size),
                "sync_symbol_errors": int(np.sum(sync_evaluated)),
                "trailer_packet_count": int(trailer_evaluated.size),
                "trailer_symbol_errors": int(np.sum(trailer_evaluated)),
            },
            arrays={
                "block_rms_devm": rms,
                "block_peak_devm": peak,
                "symbol_devm": symbol_devm,
                "omega_i_hz": omega_i,
                "omega0_hz": omega0,
                "omega_i_plus_omega0_hz": combined_frequency,
                "guard_time_s": guard_time,
                "payload_bit_errors": payload_bit_errors,
                "sync_symbol_errors": sync_bit_errors,
                "trailer_symbol_errors": trailer_bit_errors,
            },
            metadata={"required_blocks": int(required_blocks), "modulation": modulation},
        )

    def aggregate_fsk(self) -> BluetoothRFMeasurementResult:
        fsk = [result for result in self._results if result.test_case_id == "bluetooth.fsk"]
        reasons = tuple(dict.fromkeys(
            reason
            for result in fsk
            for reason in result.eligibility.reasons
        ))
        f1 = np.asarray(
            [
                float(result.metrics["delta_f1_avg_hz"])
                for result in fsk
                if result.metrics.get("delta_f1_avg_hz") is not None
            ],
            dtype=np.float64,
        )
        f2 = np.asarray(
            [
                float(result.metrics["delta_f2_avg_hz"])
                for result in fsk
                if result.metrics.get("delta_f2_avg_hz") is not None
            ],
            dtype=np.float64,
        )
        f2_max_arrays = [
            result.arrays.get("delta_f2_max_hz", np.empty(0))
            for result in fsk
            if result.metrics.get("delta_f2_avg_hz") is not None
        ]
        f2_max = (
            np.concatenate(f2_max_arrays)
            if f2_max_arrays
            else np.empty(0, dtype=np.float64)
        )
        if not f1.size:
            reasons = (*reasons, "requires at least one 11110000 packet")
        if not f2.size:
            reasons = (*reasons, "requires at least one 10101010 packet")
        eligibility = RFTestEligibility.from_reasons(reasons)
        f1_avg = float(np.mean(f1)) if f1.size else None
        f2_avg = float(np.mean(f2)) if f2.size else None
        ratio = (
            f2_avg / f1_avg
            if f1_avg is not None and f2_avg is not None and f1_avg > 0.0
            else None
        )
        f2_p999_floor = (
            float(np.percentile(f2_max, 0.1)) if f2_max.size else None
        )
        filter_profiles = {
            str(result.metadata.get("filter_profile", "")) for result in fsk
        }
        verdict = RFTestVerdict.NOT_APPLICABLE
        if eligibility.eligible and filter_profiles == {"br_1m"}:
            f1_low, f1_high = BR_DELTA_F1_AVG_RANGE_HZ
            verdict = (
                RFTestVerdict.PASS
                if f1_avg is not None
                and f1_low <= f1_avg <= f1_high
                and f2_p999_floor is not None
                and f2_p999_floor >= BR_DELTA_F2_P999_MIN_HZ
                and ratio is not None
                and ratio >= FSK_DELTA_F2_RATIO_MIN
                else RFTestVerdict.FAIL
            )
        return BluetoothRFMeasurementResult(
            "bluetooth.fsk.aggregate",
            eligibility,
            verdict,
            metrics={
                "packet_count": len(fsk),
                "delta_f1_packet_count": int(f1.size),
                "delta_f2_packet_count": int(f2.size),
                "delta_f1_avg_hz": f1_avg,
                "delta_f2_avg_hz": f2_avg,
                "delta_f2_p999_floor_hz": f2_p999_floor,
                "delta_f2_ratio": ratio,
            },
            arrays={
                "delta_f1_packet_hz": f1,
                "delta_f2_packet_hz": f2,
                "delta_f2_max_hz": f2_max,
            },
            metadata={
                "filter_profiles": tuple(sorted(filter_profiles)),
                "limits_require_phy_and_test_case_selection": verdict
                is RFTestVerdict.NOT_APPLICABLE,
            },
        )

    def aggregate_hdt(
        self, *, required_packets: int = 1500
    ) -> BluetoothRFMeasurementResult:
        """Aggregate per-packet HDT RMS EVM evidence."""

        hdt = [
            result
            for result in self._results
            if result.test_case_id == "bluetooth.hdt.evm"
        ]
        eligible = [result for result in hdt if result.eligibility.eligible]
        header_db = np.asarray(
            [float(result.metrics["header_rms_evm_db"]) for result in eligible],
            dtype=np.float64,
        )
        payload_db = np.asarray(
            [float(result.metrics["payload_rms_evm_db"]) for result in eligible],
            dtype=np.float64,
        )
        header_pass = np.asarray(
            [bool(result.metrics["header_pass"]) for result in eligible],
            dtype=bool,
        )
        payload_pass = np.asarray(
            [bool(result.metrics["payload_pass"]) for result in eligible],
            dtype=bool,
        )
        failed = np.flatnonzero(~(header_pass & payload_pass))
        count = len(eligible)
        reasons = tuple(
            dict.fromkeys(
                reason
                for result in hdt
                for reason in result.eligibility.reasons
            )
        )
        if count < int(required_packets):
            reasons = (
                *reasons,
                f"requires {required_packets} RMS EVM packets; {count} available",
            )
        eligibility = RFTestEligibility.from_reasons(reasons)
        verdict = RFTestVerdict.NOT_APPLICABLE
        if eligibility.eligible:
            verdict = (
                RFTestVerdict.PASS
                if bool(np.all(header_pass)) and bool(np.all(payload_pass))
                else RFTestVerdict.FAIL
            )
        return BluetoothRFMeasurementResult(
            "bluetooth.hdt.evm.aggregate",
            eligibility,
            verdict,
            metrics={
                "packet_count": len(hdt),
                "eligible_packet_count": count,
                "header_pass_count": int(np.count_nonzero(header_pass)),
                "payload_pass_count": int(np.count_nonzero(payload_pass)),
                "first_failure": None if not failed.size else int(failed[0]) + 1,
                "worst_header_rms_evm_db": (
                    None if not header_db.size else float(np.max(header_db))
                ),
                "worst_payload_rms_evm_db": (
                    None if not payload_db.size else float(np.max(payload_db))
                ),
            },
            arrays={
                "header_rms_evm_db": header_db,
                "payload_rms_evm_db": payload_db,
            },
            metadata={"required_packets": int(required_packets)},
        )
