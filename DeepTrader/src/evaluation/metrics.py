"""
DeepTrader — Evaluation Metrics
Financial and ML metrics for evaluating trajectory predictions.
"""

import numpy as np
from typing import Dict, List


def direction_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    What % of predicted candles got the direction right?
    Direction = whether close > open (bullish) or close < open (bearish).

    Args:
        predicted: (N, K, 17) predicted trajectories
        actual: (N, K, 17) actual trajectories
    """
    # Column 4 = body_ratio, positive = bullish
    pred_dir = predicted[:, :, 4] > 0  # body_ratio > 0 = bullish
    actual_dir = actual[:, :, 4] > 0
    return (pred_dir == actual_dir).mean()


def trajectory_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error on OHLC features (first 4 columns)."""
    return np.abs(predicted[:, :, :4] - actual[:, :, :4]).mean()


def trajectory_direction_consistency(trajectory: np.ndarray) -> float:
    """
    How consistent is the trajectory direction?
    1.0 = all candles in the same direction
    0.0 = equal mix of bullish/bearish
    """
    directions = trajectory[:, :, 4] > 0  # body_ratio > 0 = bullish
    bullish_ratio = directions.mean(axis=1)  # Per-sample consistency
    # Consistency = how far from 0.5
    consistency = np.abs(bullish_ratio - 0.5) * 2
    return consistency.mean()


def net_trajectory_move(trajectory: np.ndarray) -> np.ndarray:
    """
    Net price move of each trajectory.
    Computed from normalized close values (column 3).

    Returns: (N,) array of net moves
    """
    return trajectory[:, -1, 3] - trajectory[:, 0, 0]  # Last close - first open


def trading_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute trading-oriented metrics by treating trajectories as signals.

    A trade is taken when the predicted trajectory moves > threshold (in normalized units).
    The actual outcome determines P&L.
    """
    pred_move = net_trajectory_move(predictions)
    actual_move = net_trajectory_move(actuals)

    # Long signals: predicted upward move
    long_signals = pred_move > threshold
    # Short signals: predicted downward move
    short_signals = pred_move < -threshold
    # No trade
    no_trade = ~long_signals & ~short_signals

    n_longs = long_signals.sum()
    n_shorts = short_signals.sum()
    n_total = n_longs + n_shorts

    if n_total == 0:
        return {
            "n_trades": 0, "n_longs": 0, "n_shorts": 0,
            "win_rate": 0, "profit_factor": 0, "avg_pnl": 0,
            "sharpe": 0, "max_drawdown": 0,
        }

    # P&L for each signal
    pnl = np.zeros(len(predictions))
    pnl[long_signals] = actual_move[long_signals]      # Long: profit when price goes up
    pnl[short_signals] = -actual_move[short_signals]   # Short: profit when price goes down

    # Only traded positions
    traded_pnl = pnl[long_signals | short_signals]

    wins = traded_pnl > 0
    losses = traded_pnl < 0

    win_rate = wins.sum() / max(1, n_total)
    gross_profit = traded_pnl[wins].sum() if wins.any() else 0
    gross_loss = abs(traded_pnl[losses].sum()) if losses.any() else 1e-10
    profit_factor = gross_profit / gross_loss

    # Sharpe (annualized, assuming ~250 trading days)
    avg_pnl = traded_pnl.mean()
    std_pnl = traded_pnl.std() if len(traded_pnl) > 1 else 1e-10
    sharpe = (avg_pnl / std_pnl) * np.sqrt(250) if std_pnl > 1e-10 else 0

    # Max drawdown
    cumulative = np.cumsum(traded_pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = drawdown.max() if len(drawdown) > 0 else 0

    return {
        "n_trades": int(n_total),
        "n_longs": int(n_longs),
        "n_shorts": int(n_shorts),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_pnl": float(avg_pnl),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def evaluate_model(
    predictions: np.ndarray,
    actuals: np.ndarray,
    thresholds: List[float] = [0.3, 0.5, 0.7, 1.0, 1.5],
) -> Dict[str, any]:
    """
    Full evaluation: ML metrics + trading metrics at multiple thresholds.
    """
    results = {
        "ml_metrics": {
            "direction_accuracy": float(direction_accuracy(predictions, actuals)),
            "trajectory_mae": float(trajectory_mae(predictions, actuals)),
            "direction_consistency": float(trajectory_direction_consistency(predictions)),
        },
        "trading_metrics": {},
    }

    for thresh in thresholds:
        results["trading_metrics"][f"threshold_{thresh}"] = trading_metrics(
            predictions, actuals, threshold=thresh
        )

    return results


def print_evaluation(results: Dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'═'*60}")
    print(f"  Model Evaluation Results")
    print(f"{'═'*60}")

    print(f"\n  ML Metrics:")
    ml = results["ml_metrics"]
    print(f"    Direction accuracy:    {ml['direction_accuracy']:.2%}")
    print(f"    Trajectory MAE:        {ml['trajectory_mae']:.4f}")
    print(f"    Direction consistency: {ml['direction_consistency']:.2%}")

    print(f"\n  Trading Metrics by Threshold:")
    print(f"  {'Thresh':>8} {'Trades':>7} {'Win%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─'*50}")

    for key, tm in results["trading_metrics"].items():
        thresh = key.replace("threshold_", "")
        print(f"  {thresh:>8} {tm['n_trades']:>7} {tm['win_rate']:>5.1%} "
              f"{tm['profit_factor']:>6.2f} {tm['sharpe']:>7.2f} {tm['max_drawdown']:>8.4f}")

    print(f"\n{'═'*60}")
