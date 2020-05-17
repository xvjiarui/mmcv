# Copyright (c) Open-MMLab. All rights reserved.
from .collate import collate
from .data_container import DataContainer
from .data_parallel import MMDataParallel
from .scatter_gather import scatter, scatter_kwargs

try:
    from .distributed import MMDistributedDataParallel
except ImportError:
    from .distributed_deprecated import MMDistributedDataParallel
    import warnings
    warnings.warn('Cannot import official DistributedDataParallel and use a'
                  ' deprecated version instead.')

__all__ = [
    'collate', 'DataContainer', 'MMDataParallel', 'MMDistributedDataParallel',
    'scatter', 'scatter_kwargs'
]
