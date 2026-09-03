import torch

from MNovo.denovo.data import DeNovoDataModule


def test_epoch_sampler_is_without_replacement_and_budgeted() -> None:
    module = DeNovoDataModule(
        batch_size=8,
        n_workers=0,
        train_num_samples=50,
    )
    module.train_dataset = torch.utils.data.TensorDataset(torch.arange(100))
    first_epoch = list(module.train_dataloader().sampler)
    second_epoch = list(module.train_dataloader().sampler)

    assert len(first_epoch) == 50
    assert len(set(first_epoch)) == 50
    assert len(second_epoch) == 50
    assert len(set(second_epoch)) == 50
    assert first_epoch != second_epoch
