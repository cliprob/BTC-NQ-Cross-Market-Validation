import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESEARCH_DIR = PROJECT_ROOT / "research"

BTC_DATA_FILE = DATA_DIR / "BTCUSDT_1m_2024-03-07_to_2026-03-07.csv"
NQ_DATA_FILE = DATA_DIR / "NQ_stitched_1min_2024-03-07_to_2026-03-07.csv"

DEFAULT_ROLLING_WINDOW = 15
DEFAULT_HOLD_HORIZONS = np.arange(0, 65, 5)
DEFAULT_SENSITIVITY_WINDOWS = np.arange(0, 65, 5)
DEFAULT_SENSITIVITY_HORIZONS = np.arange(0, 65, 5)

# Test all times (24-hour coverage; CME maintenance gap already handled via inner-join data sync)
SIGNAL_SESSION_START = "00:00"
SIGNAL_SESSION_END = "23:59"
SIGNAL_TIMEZONE = "America/New_York"
DEFAULT_HIT_RATIO_THRESHOLD = 0.70
DEFAULT_SPEARMAN_THRESHOLD = 0.60
