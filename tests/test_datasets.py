import json
from pathlib import Path

import numpy as np
import pytest

from ridge_lab import make_dataset


def test_dataset_contract_fixture_is_reproducible_and_disjoint() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/dataset_contract.json").read_text())
    counts = fixture["sample_counts"]
    first = make_dataset(
        seed=fixture["seed"],
        noise_std=fixture["noise_std"],
        n_train=counts["train"],
        n_val=counts["validation"],
        n_test=counts["test"],
    )
    second = make_dataset(
        seed=fixture["seed"],
        noise_std=fixture["noise_std"],
        n_train=counts["train"],
        n_val=counts["validation"],
        n_test=counts["test"],
    )

    for name, split in (("train", first.train), ("validation", first.validation), ("test", first.test)):
        assert split.x.shape == split.y.shape == (fixture["expected_shapes"][name],)
        assert np.array_equal(split.x, getattr(second, name).x)
        assert np.array_equal(split.y, getattr(second, name).y)
        np.testing.assert_allclose(split.x, fixture["sample"][name]["x"])
        np.testing.assert_allclose(split.y, fixture["sample"][name]["y"])
        assert list(split.sample_ids) == fixture["sample"][name]["sample_ids"]
        assert not split.x.flags.writeable
    ids = first.train.sample_ids + first.validation.sample_ids + first.test.sample_ids
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("kwargs", [{"n_train": 0}, {"n_val": -1}, {"noise_std": -0.1}])
def test_dataset_rejects_invalid_settings(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_dataset(**kwargs)
