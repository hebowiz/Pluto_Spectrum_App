from pluto_sa.vsa.result_statistics import ResultSummaryAccumulator


def test_all_packet_statistics_use_measurement_appropriate_averages() -> None:
    statistics = ResultSummaryAccumulator()

    statistics.add(
        {
            "modulation": "QPSK",
            "result_symbols": "100",
            "power": "+0.00 dBm",
            "evm_rms": "3.00 % (-30.46 dB)",
            "carrier_frequency_error": "+1.000 kHz",
            "pattern_symbols_correct": "Yes",
        }
    )
    statistics.add(
        {
            "modulation": "QPSK",
            "result_symbols": "200",
            "power": "+10.00 dBm",
            "evm_rms": "4.00 % (-27.96 dB)",
            "carrier_frequency_error": "-3.000 kHz",
            "pattern_symbols_correct": "No",
        }
    )

    values = statistics.values()
    assert values["power"] == "+8.45 [+0.00 … +10.00] dBm (N=2)"
    assert values["evm_rms"] == "3.70 [3.00 … 4.00] % (N=2)"
    assert values["carrier_frequency_error"] == "-1.000 [-3.000 … +1.000] kHz (N=2)"
    assert values["result_symbols"] == "150.0 [100.0 … 200.0] symbols (N=2)"
    assert values["pattern_symbols_correct"] == "Yes 1/2"
    assert values["modulation"] == "QPSK: 2 (N=2)"
    assert values["match_selection"] == "2 packet(s)"


def test_all_packet_statistics_reset_releases_prior_measurements() -> None:
    statistics = ResultSummaryAccumulator()
    statistics.add({"power": "-30.00 dBm", "result_symbols": "10"})

    statistics.clear()

    assert statistics.packet_count == 0
    assert statistics.values()["match_selection"] == "0 packet(s)"
