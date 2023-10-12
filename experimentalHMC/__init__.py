from .__version import __version__
from .onlinemean import OnlineMean
from .onlinemeanvar import OnlineMeanVar
from .onlinequantile import OnlineQuantile
from .onlinemad import OnlineMAD

from .leapfrog import leapfrog

__all__ = [
    "OnlineMean",
    "OnlineMeanVar",
    "OnlineQuantile",
    "OnlineMAD",
    "leapfrog",
]
