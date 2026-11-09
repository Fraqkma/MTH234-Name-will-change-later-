from ridge_lab import DatasetBundle, DatasetSplit, make_dataset


def test_package_exports_contract_and_default_split_sizes() -> None:
    dataset = make_dataset()
    assert isinstance(dataset, DatasetBundle)
    assert isinstance(dataset.train, DatasetSplit)
    assert (len(dataset.train.x), len(dataset.validation.x), len(dataset.test.x)) == (24, 24, 120)
