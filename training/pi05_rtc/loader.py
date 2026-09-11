"""Official v3 image loading with explicit retained-row action windows."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# pylint: disable=import-outside-toplevel


from dataclasses import dataclass
from collections.abc import Callable

from training.common.dataset import ExportedDataset, rgb_images
from training.pi05_rtc import transforms


@dataclass
class Samples:
    """A spawn-safe dataset view; decoding occurs only for requested samples."""

    dataset: ExportedDataset
    transform: Callable[[dict], dict]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset.official()[index]
        images = rgb_images(row)
        item = transforms.inputs(
            self.dataset.states[index],
            images,
            row["task"],
            self.dataset.window(index, 50),
        )
        return self.transform(item)


def create(config, dataset):
    """Retain inherited batching, sharding and model preprocessing."""
    from openpi import (
        transforms as ops,
    )
    from openpi.training.data_loader import (
        DataLoaderImpl,
        TorchDataLoader,
    )

    data = config.data.create(config.assets_dirs, config.model)
    transform = ops.compose(
        [
            ops.Normalize(data.norm_stats, use_quantiles=True),
            *data.model_transforms.inputs,
        ]
    )
    samples = Samples(dataset, transform)
    return DataLoaderImpl(
        data,
        TorchDataLoader(
            samples,
            local_batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            seed=config.seed,
        ),
    )
