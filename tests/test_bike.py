"""Checks for the approved Capital Bikeshare schema and time boundaries."""

from dataclasses import replace

import numpy as np
import pytest

from ridge_lab import bike_experiment
from ridge_lab.bike import DEFAULT_HOUR_CSV, HOUR_CSV_SHA256, load_bike_hourly, make_bike_split
from ridge_lab.bike_experiment import (
    evaluate_bike_development,
    refit_bike_development,
    select_bike_settings,
)
from ridge_lab.features import make_bike_design
from ridge_lab.solvers import fit_ols, lasso


@pytest.fixture(scope="module")
def bike():  # type: ignore[no-untyped-def]
    data = load_bike_hourly()
    return data, make_bike_split(data)


@pytest.fixture(scope="module")
def bike_selection(bike):  # type: ignore[no-untyped-def]
    data, split = bike
    return select_bike_settings(evaluate_bike_development(data, split))


def test_loader_has_only_approved_values_and_stable_ids(bike) -> None:  # type: ignore[no-untyped-def]
    data, _ = bike
    assert data.sha256 == HOUR_CSV_SHA256
    assert len(data.row_ids) == len(set(data.row_ids)) == 17_379
    assert len(set(data.dates)) == 731
    assert data.row_ids[0] == "2011-01-01:00"
    assert data.row_ids[-1] == "2012-12-31:23"
    assert not data.counts.flags.writeable
    assert not data.hours.flags.writeable
    assert not data.working_days.flags.writeable
    assert not hasattr(data, "casual") and not hasattr(data, "registered")


def test_chronological_folds_keep_dates_together_and_holdout_sealed(bike) -> None:  # type: ignore[no-untyped-def]
    data, split = bike
    assert (len(split.development), len(split.holdout)) == (13_915, 3_464)
    assert split.holdout_start.isoformat() == "2012-08-08"
    assert split.holdout_end.isoformat() == "2012-12-31"
    assert len(split.folds) == 5
    assert len(set(split.development) & set(split.holdout)) == 0
    assert set(split.development) | set(split.holdout) == set(range(len(data.row_ids)))
    assert [len(fold.train) for fold in split.folds] == [2259, 4610, 6945, 9266, 11587]
    assert [len(fold.validation) for fold in split.folds] == [2351, 2335, 2321, 2321, 2328]
    for fold in split.folds:
        train_dates = {data.dates[index] for index in fold.train}
        validation_dates = {data.dates[index] for index in fold.validation}
        holdout_dates = {data.dates[index] for index in split.holdout}
        assert train_dates.isdisjoint(validation_dates | holdout_dates)
        assert validation_dates.isdisjoint(holdout_dates)
        assert max(train_dates) < min(validation_dates) < min(holdout_dates)
        assert max(train_dates) == fold.train_end
        assert min(validation_dates) == fold.validation_start


def test_bike_design_has_one_intercept_and_nonredundant_interaction(bike) -> None:  # type: ignore[no-untyped-def]
    data, split = bike
    design = make_bike_design(data.hours[list(split.folds[0].train)],
                              data.working_days[list(split.folds[0].train)])
    assert design.shape == (2259, 48)
    assert np.array_equal(design[:, 0], np.ones(2259))
    assert np.linalg.matrix_rank(design) == 48
    sample = make_bike_design(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert np.array_equal(sample[:, 0], np.ones(4))
    assert np.array_equal(sample[:, 1], [0, 0, 1, 1])  # hour 1 main effect
    assert np.array_equal(sample[:, 24], [0, 1, 0, 1])  # working-day main effect
    assert np.array_equal(sample[:, 25], [0, 0, 0, 1])  # hour 1 interaction


def test_lasso_zero_equals_ols_and_penalty_spares_intercept() -> None:
    design = make_bike_design(np.tile(np.arange(24), 4), np.repeat([0, 1], 48))
    targets = np.full(design.shape[0], 7.0)
    assert np.allclose(lasso(design, targets, 0.0), fit_ols(design, targets))
    strong = lasso(design, targets, 100.0)
    assert strong[0] == pytest.approx(7.0)
    assert np.allclose(strong[1:], 0.0)


def test_lasso_nonzero_penalty_soft_thresholds_a_known_slope() -> None:
    design = np.column_stack((np.ones(3), [-1.0, 0.0, 1.0]))
    targets = np.array([1.0, 3.0, 5.0])
    assert np.allclose(lasso(design, targets, 1.0 / 3.0), [3.0, 1.5])


def test_loader_rejects_changed_source_bytes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    changed = tmp_path / "hour.csv"
    changed.write_bytes(DEFAULT_HOUR_CSV.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_bike_hourly(changed)


def test_paired_development_comparison_does_not_read_holdout_targets(bike) -> None:  # type: ignore[no-untyped-def]
    data, split = bike
    counts = data.counts.copy()
    counts[list(split.holdout)] = np.nan
    sealed = replace(data, counts=counts)
    results = evaluate_bike_development(
        sealed, split, ridge_penalties=(0.0,), lasso_penalties=(0.1,)
    )
    assert [result.method for result in results] == ["OLS", "Ridge", "Lasso"]
    assert all(len(result.windows) == 5 for result in results)
    assert all(np.isfinite(result.mean_rmse) for result in results)


def test_refit_passes_exact_development_rows_and_reviewed_settings(
    bike, bike_selection, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    data, split = bike
    assert {method: choice.penalty for method, choice in bike_selection.items()} == {
        "OLS": 0.0, "Ridge": 0.0, "Lasso": 0.001,
    }
    counts = data.counts.copy()
    counts[list(split.holdout)] = np.nan
    sealed = replace(data, counts=counts)
    indices = np.asarray(split.development)
    expected_design = make_bike_design(data.hours[indices], data.working_days[indices])
    expected_targets = data.counts[indices]
    calls = []

    def spy(method: str):
        def fit(design, targets, *penalty):
            assert design.shape == (13_915, 48)
            np.testing.assert_array_equal(design, expected_design)
            np.testing.assert_array_equal(targets, expected_targets)
            calls.append((method, penalty))
            return np.zeros(48)

        return fit

    monkeypatch.setattr(bike_experiment, "fit_ols", spy("OLS"))
    monkeypatch.setattr(bike_experiment, "ridge", spy("Ridge"))
    monkeypatch.setattr(bike_experiment, "lasso", spy("Lasso"))
    refit = refit_bike_development(sealed, split, bike_selection)

    assert calls == [("OLS", ()), ("Ridge", (0.0,)), ("Lasso", (0.001,))]
    assert refit.training_row_ids == tuple(data.row_ids[index] for index in split.development)
    assert refit.training_start.isoformat() == "2011-01-01"
    assert refit.training_end.isoformat() == "2012-08-07"
    assert refit.source_sha256 == HOUR_CSV_SHA256
    assert all(not model.coefficients.flags.writeable for model in refit.models)


def test_refit_rejects_holdout_membership_and_changed_settings(
    bike, bike_selection, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    data, split = bike
    monkeypatch.setattr(
        bike_experiment, "fit_ols",
        lambda *_: pytest.fail("no solver should run for invalid refit input"),
    )
    contaminated = replace(
        split, development=split.development[:-1] + (split.holdout[0],)
    )
    with pytest.raises(ValueError, match="13,915 approved development rows"):
        refit_bike_development(data, contaminated, bike_selection)
    changed = dict(bike_selection)
    changed["Ridge"] = replace(changed["Ridge"], penalty=1.0)
    with pytest.raises(ValueError, match="Ridge setting differs"):
        refit_bike_development(data, split, changed)


@pytest.mark.parametrize("hours,working", [([24], [0]), ([0], [2]), ([0, 1], [1])])
def test_bike_design_rejects_bad_values(hours: list[int], working: list[int]) -> None:
    with pytest.raises(ValueError):
        make_bike_design(np.array(hours), np.array(working))


def test_normal_entry_point_masks_reserved_targets():
    from ridge_lab.bike_experiment import bike_development_payload

    data = load_bike_hourly()
    split = make_bike_split(data)
    assert np.isnan(data.counts[list(split.holdout)]).all()
    assert np.isfinite(data.counts[list(split.development)]).all()
    payload = bike_development_payload()
    assert payload["training_rows"] == 13915
    assert payload["training_end"] == "2012-08-07"
    assert [r["penalty"] for r in payload["results"]] == [0, 0, 0.001]
