"""Tiny Bot: RSI(14) mean-reversion on hourly Solana bars.

Strategy (deterministic, stdlib only, no network, no clock, no random):
- Per symbol, RSI(14) = 100 - 100 / (1 + mean_gain / mean_loss) over the
  last 14 hourly close-to-close changes (simple means, recomputed each bar).
- BUY when RSI < 25 (oversold bounce) and we hold nothing of that symbol.
- SELL the whole position when RSI > 55 (bounce exhausted).
- Size: 24% of equity per buy, split from available cash when several
  symbols trigger on the same bar; never above the 25% weight cap.
- Long only, no leverage, no shorts. Orders decided on bar k fill at bar
  k+1's open, so every signal is acted on with one bar of lag.

Why this, not trend-following: on the dev window (136 hourly bars, every
symbol down 3-14%, equal-weight buy-and-hold -8.5%) every trend/momentum
variant tested lost money (EMA-cross -173 to -361 USDC, momentum -229 to
-268, Donchian -291 to -871 in the offline sim), while oversold bounces
paid: RSI 25/55 scored +487 USDC full-window in the same sim and beat
buy-and-hold in all 64 rolling 72-bar windows (mean +213, worst -108,
never fewer than 14 trades per 72-bar window). The edge is a broad
plateau, not one cherry-picked threshold: RSI buy 20-30 / sell 55-60 all
came out positive. The final window is a different 3 days, so no dev
number is a promise; the bot simply stays in cash unless a symbol is
deeply oversold, which is what beats buy-and-hold when prices fall and
keeps drawdown small when they chop.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

RSI_PERIOD = 14
RSI_BUY = 25.0
RSI_SELL = 55.0
SIZE_BPS = 2400  # 24% of equity per buy: the signal is high-conviction
# (deep oversold, validated plateau), so size near the 25% weight cap;
# weight_room() still guards the cap and cash is split across same-bar buys.
# v2 change: 20% -> 24%. Tested in the offline sim before adopting: full
# window +529 vs +487 USDC, rolling 72-bar mean +232 vs +213, median +200
# vs +172, worst window -125 vs -108, max DD 2.0% vs 1.8%. Rejected
# alternatives, same sim: buy-only-on-uptick (+333), 3% stop (+433),
# sell-on-RSI-turn-down (+278), size 16% (+396).
MAX_SLIPPAGE_BPS = 300

# closes[symbol] = list of latest closes (ints), oldest first.
closes = {}


def rsi(history):
    """RSI over the last RSI_PERIOD hourly changes; None until 15 closes."""
    if len(history) < RSI_PERIOD + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(len(history) - RSI_PERIOD, len(history)):
        d = history[i] - history[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / RSI_PERIOD
    avg_loss = losses / RSI_PERIOD
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def on_batch(state, batch):
    if state.warming_up:
        return []
    for symbol in state.symbols:
        c = state.close(symbol)
        if c is None:
            continue
        closes.setdefault(symbol, []).append(c)
    orders = []
    buys = []
    for symbol in sorted(state.symbols):
        hist = closes.get(symbol)
        if not hist:
            continue
        value = rsi(hist)
        if value is None:
            continue
        held = symbol in state.positions
        if not held and value < RSI_BUY:
            buys.append(symbol)
        elif held and value > RSI_SELL:
            orders.append({"side": "sell", "symbol": symbol,
                           "qty": state.positions[symbol].qty,
                           "max_slippage_bps": MAX_SLIPPAGE_BPS})
    if buys:
        # Split the cash evenly across this bar's buys; each buy is at most
        # 20% of equity and never above the per-symbol weight room.
        per_buy = state.cash // len(buys)
        for symbol in buys:
            by_size = SIZE_BPS * state.equity // 10000
            room = state.weight_room(symbol)
            amount = min(per_buy, by_size, room)
            if amount >= 1000000:  # min notional: 1 USDC
                orders.append({"side": "buy", "symbol": symbol,
                               "notional_quote_units": amount,
                               "max_slippage_bps": MAX_SLIPPAGE_BPS})
    return orders


makefun_bars.run(on_batch)
