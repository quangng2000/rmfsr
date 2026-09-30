"""Step-indexed CPU augmentation: prefetching cannot change resumed samples."""
import numpy as np
import torch
from .data import SpeechPairs


class IndexedBatches(torch.utils.data.Dataset):
    def __init__(self, manifest, kwargs, seed, batch_size, first, steps):
        self.manifest, self.kwargs = manifest, kwargs
        self.seed, self.batch_size = seed, batch_size
        self.first, self.steps, self.pairs = first, steps, None

    def __len__(self):
        return max(0, self.steps - self.first)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.pairs is None:
            self.pairs = SpeechPairs(self.manifest, seed=self.seed, **self.kwargs)
        self.pairs.rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.first + index]))
        return self.pairs.batch(self.batch_size)


def worker_init(_):
    torch.set_num_threads(1)


def batch_loader(manifest, kwargs, cfg, first):
    data = IndexedBatches(manifest, kwargs, cfg['seed'], cfg['batch_size'], first, cfg['steps'])
    workers = cfg.get('num_workers', 0)
    options = dict(batch_size=None, num_workers=workers, worker_init_fn=worker_init,
                   generator=torch.Generator().manual_seed(cfg['seed'] + 10000),
                   pin_memory=cfg['device'] == 'cuda')
    if workers:
        options.update(prefetch_factor=cfg.get('prefetch_factor', 2),
                       multiprocessing_context='spawn', persistent_workers=True)
    return torch.utils.data.DataLoader(data, **options)
