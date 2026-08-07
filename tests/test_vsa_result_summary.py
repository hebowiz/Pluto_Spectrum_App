import pytest

from pluto_sa.vsa.result_summary import (
    DEFAULT_RESULT_SUMMARY_IDS,
    RESULT_SUMMARY_BY_ID,
    RESULT_SUMMARY_ITEMS,
    ResultSummaryCategory,
    normalize_result_summary_ids,
)


def test_result_summary_registry_has_stable_unique_ids_and_planned_items() -> None:
    item_ids = [item.item_id for item in RESULT_SUMMARY_ITEMS]
    assert len(item_ids) == len(set(item_ids)) == len(RESULT_SUMMARY_BY_ID)
    assert DEFAULT_RESULT_SUMMARY_IDS
    assert all(RESULT_SUMMARY_BY_ID[item_id].implemented for item_id in DEFAULT_RESULT_SUMMARY_IDS)
    assert any(
        item.category is ResultSummaryCategory.PSK and not item.implemented
        for item in RESULT_SUMMARY_ITEMS
    )
    assert any(
        item.category is ResultSummaryCategory.FSK and not item.implemented
        for item in RESULT_SUMMARY_ITEMS
    )


def test_result_summary_persistence_normalizes_old_and_future_settings() -> None:
    assert normalize_result_summary_ids(None) == set(DEFAULT_RESULT_SUMMARY_IDS)
    assert normalize_result_summary_ids(
        ["power", "frequency_fit_rms", "future-result"]
    ) == {"power", "frequency_fit_rms"}
    assert normalize_result_summary_ids(["evm_peak"]) == set()

    with pytest.raises(ValueError, match="array of strings"):
        normalize_result_summary_ids("power")
    with pytest.raises(ValueError, match="array of strings"):
        normalize_result_summary_ids(["power", 123])
