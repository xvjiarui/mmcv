# Copyright (c) Open-MMLab. All rights reserved.
from .collate import collate
from .data_container import DataContainer
from .data_parallel import MMDataParallel
try:
    from .distributed import MMDistributedDataParallel
except:
    from .distributed_deprecated import MMDistributedDataParallel
    import warnings
    warnings.warn('Cannot import official DistributedDataParallel and use a'
                  ' deprecated version instead.')
from .scatter_gather import scatter, scatter_kwargs

__all__ = [
    'collate', 'DataContainer', 'MMDataParallel', 'MMDistributedDataParallel',
    'scatter', 'scatter_kwargs'
]
