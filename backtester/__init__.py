"""Event-driven backtesting engine."""

from .data import (DataHandler, PandasDataHandler, generate_synthetic, load_alpaca, load_csv,
                   load_yfinance)
from .engine import Backtest, BacktestResult
from .events import Direction, FillEvent, MarketEvent, OrderEvent, SignalEvent
from .execution import (FixedBpsSlippage, NoCommission, NoSlippage, PercentCommission,
                        PerShareCommission, SimulatedExecutionHandler, VolumeShareSlippage)
from .metrics import compute_metrics, relative_metrics
from .portfolio import Portfolio
from .report import comparison_table, run_benchmark, text_report
from .risk_free import fetch_risk_free
from .strategy import (STRATEGIES, BollingerReversionStrategy, BuyAndHoldStrategy,
                       MomentumStrategy, MovingAverageCrossStrategy, Strategy)
from .walk_forward import WalkForwardResult, walk_forward
