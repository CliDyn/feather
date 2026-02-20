"""Feather — Lightweight Climate Model Evaluation Framework."""

__version__ = "0.1.0"

from feather.config import FeatherConfig
from feather.data.cmip6 import CMIP6Loader
from feather.data.loader import DataLoader
from feather.data.obs import ObsLoader
from feather.llm.analyzer import FigureAnalyzer
