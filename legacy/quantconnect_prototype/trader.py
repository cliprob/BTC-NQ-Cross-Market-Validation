# region imports
from AlgorithmImports import *
from scipy.stats import spearmanr
from datetime import time as dtime, timedelta
import numpy as np
# endregion

class NasdaqMicroFuturesTrader(QCAlgorithm):
    """
    Entry  : BTC/NQ co-movement regime
               Rolling window  = 55 bars
               Hit ratio       ≥ 0.80  (% of bars BTC and NQ moved same direction)
               Spearman corr   ≥ 0.70  (over the same 55-bar window)
               Session         : NY only  09:30 – 15:55 ET

    Direction: determined at entry from NQ momentum (long if last bar up, short if last bar down)

    Exit priority (first condition that fires wins):
        1. B2 momentum reversal — exit when NQ has moved ≥ 5 bps AGAINST the
           position over the last 7 bars.  This was the highest-Sharpe exit
           found in research (Sharpe 3.11 vs benchmark 1.25).
        2. Fixed hold fallback  — exit after 35 minutes if B2 never fires.
        3. EOD hard close       — exit any open position at 15:55 ET.
    """

    def Initialize(self):

        # ── Session boundaries (ET) ───────────────────────────────────
        self.SESSION_START = dtime(9, 30)
        self.EOD_CLOSE     = dtime(15, 55)

        # ── Backtest window ───────────────────────────────────────────
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2026, 4, 23)
        self.SetCash(100_000)
        self.SetTimeZone(TimeZones.NewYork)
        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin,
        )

        # ── Securities ────────────────────────────────────────────────
        self.btc = self.AddCrypto(
            "BTCUSDT", Resolution.Minute, Market.Binance
        ).Symbol

        nq_future = self.AddFuture(
            Futures.Indices.MicroNASDAQ100EMini,
            Resolution.Minute,
            dataNormalizationMode=DataNormalizationMode.Raw,
        )
        nq_future.SetFilter(timedelta(0), timedelta(182))
        self.nq_cont = nq_future

        # ── Entry parameters (from robustness grid search) ────────────
        self._window_size          = 55     # rolling window (bars)
        self._hit_ratio_threshold  = 0.80
        self._spearman_threshold   = 0.70

        # ── Exit parameters ───────────────────────────────────────────
        # B2 momentum reversal (best exit from research)
        self._b2_k              = 7      # look back this many bars
        self._b2_threshold_bps  = -5.0   # exit when directional move < -5 bps

        # Fixed hold fallback
        self._hold_minutes = 35

        # ── Rolling price windows ─────────────────────────────────────
        # window_size + 1 so we always have a full set of *returns*
        # (55 returns require 56 prices) and so that _mnq_prices[b2_k-1]
        # is available from the very first bar after the window fills.
        self._mnq_prices = RollingWindow[float](self._window_size + 1)
        self._btc_prices = RollingWindow[float](self._window_size + 1)

        # ── Trade state ───────────────────────────────────────────────
        self._entry_time        = None
        self._entry_price       = None
        self._trade_direction   = 0      # +1 = long, -1 = short

    # ─────────────────────────────────────────────────────────────────
    # DATA HANDLER
    # ─────────────────────────────────────────────────────────────────
    def OnData(self, data: Slice):
        current_time = self.Time.time()

        # Hard EOD close — regardless of any other condition
        if self.Portfolio.Invested and current_time >= self.EOD_CLOSE:
            self._exit_position("EOD hard close")
            return

        # Outside session — do nothing
        if not self._in_session(current_time):
            return

        # Feed rolling windows
        if data.ContainsKey(self.btc):
            self._btc_prices.Add(data[self.btc].Close)

        mapped = self.nq_cont.Mapped
        if mapped and data.ContainsKey(mapped):
            self._mnq_prices.Add(data[mapped].Close)

        # Need full windows before any logic runs
        if not (self._mnq_prices.IsReady and self._btc_prices.IsReady):
            return

        self._trading_logic(mapped)

    # ─────────────────────────────────────────────────────────────────
    # CORE LOGIC
    # ─────────────────────────────────────────────────────────────────
    def _trading_logic(self, mapped_symbol):
        if not mapped_symbol:
            return

        # ── Exit checks (run first so we don't re-enter on same bar) ──
        if self.Portfolio.Invested:
            # 1. B2 momentum reversal exit
            if self._b2_exit_triggered():
                self._exit_position("B2 momentum reversal")
                return

            # 2. Fixed hold fallback
            if self._entry_time and (self.Time - self._entry_time) >= timedelta(minutes=self._hold_minutes):
                self._exit_position(f"Hold time {self._hold_minutes}m exceeded")
                return

        # ── Entry check ───────────────────────────────────────────────
        if not self.Portfolio.Invested:
            hit_ratio = self._hit_ratio()
            spearman  = self._spearman()

            if hit_ratio >= self._hit_ratio_threshold and spearman >= self._spearman_threshold:
                # Direction: follow NQ momentum on the entry bar
                direction = int(np.sign(self._mnq_prices[0] - self._mnq_prices[1]))
                if direction == 0:
                    direction = 1   # flat bar — default long

                self._enter_position(mapped_symbol, direction, hit_ratio, spearman)

    # ─────────────────────────────────────────────────────────────────
    # B2 EXIT
    # ─────────────────────────────────────────────────────────────────
    def _b2_exit_triggered(self) -> bool:
        """
        True when NQ has moved more than b2_threshold_bps AGAINST the
        position over the last b2_k bars.

        roll_bps = (price_now / price_k_bars_ago - 1) * direction * 10_000
        Exit fires when roll_bps < b2_threshold_bps  (i.e. < -5).

        RollingWindow index 0 = most recent bar,
                        index k-1 = k bars ago.
        """
        k = self._b2_k
        # Guard: window must contain at least k+1 prices
        if self._mnq_prices.Count < k + 1:
            return False

        price_now     = self._mnq_prices[0]
        price_k_ago   = self._mnq_prices[k]   # index k = k bars ago (0-based)

        if price_k_ago == 0:
            return False

        roll_bps = (price_now / price_k_ago - 1.0) * self._trade_direction * 10_000
        return roll_bps < self._b2_threshold_bps

    # ─────────────────────────────────────────────────────────────────
    # ENTRY / EXIT WRAPPERS
    # ─────────────────────────────────────────────────────────────────
    def _enter_position(self, mapped_symbol, direction: int, hit_ratio: float, spearman: float):
        quantity = direction * 1   # +1 contract long or -1 contract short
        self.MarketOrder(mapped_symbol, quantity)

        self._entry_time      = self.Time
        self._entry_price     = self._mnq_prices[0]
        self._trade_direction = direction

        side = "LONG" if direction == 1 else "SHORT"
        self.Debug(
            f"{self.Time} | ENTRY {side} | "
            f"price={self._entry_price:.2f} | "
            f"HR={hit_ratio:.3f} | SP={spearman:.3f}"
        )

    def _exit_position(self, reason: str = ""):
        mapped = self.nq_cont.Mapped
        if not mapped or not self.Portfolio.Invested:
            return

        exit_price = self._mnq_prices[0] if self._mnq_prices.Count > 0 else 0
        pnl_pts    = (exit_price - self._entry_price) * self._trade_direction if self._entry_price else 0
        hold_secs  = int((self.Time - self._entry_time).total_seconds()) if self._entry_time else 0

        self.Liquidate(mapped)

        self.Debug(
            f"{self.Time} | EXIT ({reason}) | "
            f"price={exit_price:.2f} | "
            f"pnl_pts={pnl_pts:.2f} | "
            f"held={hold_secs // 60}m{hold_secs % 60:02d}s"
        )

        self._entry_time      = None
        self._entry_price     = None
        self._trade_direction = 0

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR CALCULATIONS
    # ─────────────────────────────────────────────────────────────────
    def _hit_ratio(self) -> float:
        """
        Fraction of the last window_size bars where BTC and NQ moved
        in the same direction.  Uses window_size return pairs, so needs
        window_size + 1 prices — which is exactly what the rolling windows hold.
        """
        n = self._window_size
        mnq = [self._mnq_prices[i] for i in range(n, -1, -1)]   # oldest → newest
        btc = [self._btc_prices[i] for i in range(n, -1, -1)]

        matches = sum(
            1 for i in range(1, n + 1)
            if np.sign(mnq[i] - mnq[i - 1]) == np.sign(btc[i] - btc[i - 1])
        )
        return matches / n

    def _spearman(self) -> float:
        """
        Spearman correlation of 1-minute returns over the last window_size bars.
        """
        n = self._window_size
        mnq = [self._mnq_prices[i] for i in range(n, -1, -1)]
        btc = [self._btc_prices[i] for i in range(n, -1, -1)]

        mnq_rets = [(mnq[i] - mnq[i - 1]) / mnq[i - 1] for i in range(1, n + 1) if mnq[i - 1] != 0]
        btc_rets = [(btc[i] - btc[i - 1]) / btc[i - 1] for i in range(1, n + 1) if btc[i - 1] != 0]

        if len(mnq_rets) < 10 or len(btc_rets) < 10:
            return 0.0

        try:
            corr, _ = spearmanr(mnq_rets, btc_rets)
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────
    def _in_session(self, t) -> bool:
        return self.SESSION_START <= t < self.EOD_CLOSE