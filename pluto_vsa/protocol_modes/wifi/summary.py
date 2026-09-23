"""Presentation of typed RF/PHY, decode and diagnostic results."""
from dataclasses import replace
from .measurements import measurement_results


def visible_results(result, recording, *, diagnostics=False, statistics=None, conditions=None):
    measurements = (measurement_results(result,recording,statistics or {},conditions=conditions)
                    if conditions is not None else result.measurements or measurement_results(result,recording,{}))
    # IQ Power's existing display-plane selection also selects packet power.
    values = {"packet_power":result.packet_power_dbm,"peak_power":result.peak_power_dbm}
    return tuple(replace(item,value=values[item.measurement_id]) if item.measurement_id in values else item
                 for item in measurements if diagnostics or item.default_visible)


def summary_rows(result, recording, *, diagnostics=False):
    return [item.row() for item in visible_results(result,recording,diagnostics=diagnostics)]
