import numpy as np

from pluto_sa.vsa.protocol_modes.dect import analyze_dect_recording
from pluto_sa.vsa.sources import FileIQSource
from tools.generate_dect_iq import main


def test_generate_dect_iq_command_writes_analyzable_npz(tmp_path) -> None:
    output = tmp_path / "dect_pp_p80.npz"
    assert main(
        [
            "--output",
            str(output),
            "--direction",
            "PP",
            "--packet",
            "P80",
            "--frequency-error-hz",
            "9000",
        ]
    ) == 0
    with np.load(output, allow_pickle=False) as container:
        assert container["generated_bits"].size == 900
    result = analyze_dect_recording(FileIQSource.load(output))[0]
    assert result.direction == "PP"
    assert result.packet_type == "P80"
    assert result.packet_analysis.integrity.crc_valid is True
    assert next(
        item for item in result.packet_analysis.summary if item.key == "x_crc"
    ).display == "PASS"


def test_generate_dect_iq_command_supports_prolonged_preamble(tmp_path) -> None:
    output = tmp_path / "dect_pp_p32_prolonged.npz"
    assert main(
        [
            "--output",
            str(output),
            "--direction",
            "PP",
            "--prolonged-preamble",
        ]
    ) == 0
    with np.load(output, allow_pickle=False) as container:
        assert container["generated_bits"].size == 436
        assert str(container["preamble_mode"].item()) == "Prolonged"
    result = analyze_dect_recording(FileIQSource.load(output))[0]
    assert result.preamble_mode == "Prolonged"
    assert result.direction == "PP"
