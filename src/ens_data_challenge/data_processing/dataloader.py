from torch.utils.data import DataLoader


def get_train_dataloader(dataset, batch_size: int = 42, num_workers: int = 4, sampler=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader


def get_val_dataloader(val_dataset, batch_size: int = 42, num_workers: int = 4):
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return loader
