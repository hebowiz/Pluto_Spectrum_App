import numpy as np
from pluto_vsa.mapping import reverse_symbol_bits
from pluto_vsa.pattern import KnownPattern


def _pattern_from_generated(recording, start: int, length: int) -> KnownPattern:
    symbols = np.asarray(recording.metadata["generated_symbols"])
    maximum = int(np.max(symbols))
    order = 2 if maximum < 2 else (4 if maximum < 4 else 8)
    displayed = reverse_symbol_bits(symbols[start : start + length], order)
    return KnownPattern(tuple(int(value) for value in displayed))
