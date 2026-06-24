"""Benford's Law digit-distribution analysis for transaction amounts.

Computes the chi-square statistic, per-digit Z-scores, and Mean Absolute
Deviation (MAD) of the leading-digit distribution of a set of amounts,
relative to the theoretical Benford distribution.

The univariate helpers (`compute_benford_metrics` etc.) score a single
`(wallet, asset_pair)` stream. A coordinated wash-trading syndicate can keep
each individual pair close to Benford while the *joint* cross-pair behaviour is
statistically impossible under independent trading. The multivariate helpers
(`joint_digit_matrix`, `benford_copula_statistic`, `cross_pair_sync_score`,
`multivariate_benford_score`) surface that coordination signal.

Adaptive window sizing (``AdaptiveBenfordWindow``) ensures that Benford metrics
are only computed when the sample count N >= ``BENFORD_MIN_SAMPLE_COUNT``. When
a target window contains fewer trades, the window is doubled up to
``BENFORD_MAX_WINDOW_DAYS``. If expansion still cannot reach the minimum, the
result is marked ``valid=False`` so downstream consumers can handle
statistically unreliable windows gracefully.
"""

import bisect
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

logger = logging.getLogger(__name__)

DIGITS = list(range(1, 10))

# P(d) = log10(1 + 1/d) for d in 1..9
BENFORD_EXPECTED: dict[int, float] = {d: math.log10(1 + 1 / d) for d in DIGITS}

# Entropy (nats) of the theoretical Benford leading-digit distribution.
BENFORD_ENTROPY: float = float(-sum(p * math.log(p) for p in BENFORD_EXPECTED.values()))


def first_digit(value: float) -> int | None:
    """Return the leading (most significant) decimal digit of `value`.

    Returns None for zero, negative, or non-finite values, which are
    excluded from Benford analysis.
    """
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    while value < 1:
        value *= 10
    while value >= 10:
        value /= 10
    return int(value)


def digit_distribution(amounts: list[float]) -> dict[int, float]:
    """Return the observed proportion of each leading digit 1-9 in `amounts`."""
    digits = [d for d in (first_digit(a) for a in amounts) if d is not None]
    n = len(digits)
    if n == 0:
        return {d: 0.0 for d in DIGITS}
    counts = {d: 0 for d in DIGITS}
    for d in digits:
        counts[d] += 1
    return {d: counts[d] / n for d in DIGITS}


def chi_square_statistic(observed: dict[int, float], n: int) -> float:
    """Chi-square goodness-of-fit statistic vs. the Benford distribution.

    `observed` is a digit -> proportion mapping (e.g. from `digit_distribution`).
    `n` is the number of observations the proportions were computed from.
    """
    if n == 0:
        return 0.0
    chi_sq = 0.0
    for d in DIGITS:
        expected_count = BENFORD_EXPECTED[d] * n
        observed_count = observed.get(d, 0.0) * n
        if expected_count > 0:
            chi_sq += (observed_count - expected_count) ** 2 / expected_count
    return chi_sq


def z_scores(observed: dict[int, float], n: int) -> dict[int, float]:
    """Per-digit Z-score of the observed proportion vs. Benford's expectation."""
    if n == 0:
        return {d: 0.0 for d in DIGITS}
    scores = {}
    for d in DIGITS:
        p = BENFORD_EXPECTED[d]
        observed_p = observed.get(d, 0.0)
        # continuity correction as commonly used in Benford forensic analysis
        numerator = abs(observed_p - p) - (1 / (2 * n))
        denominator = math.sqrt(p * (1 - p) / n)
        scores[d] = max(numerator, 0.0) / denominator if denominator > 0 else 0.0
    return scores


def mean_absolute_deviation(observed: dict[int, float]) -> float:
    """MAD between observed and expected digit distributions.

    Values above ~0.015 (for first-digit tests) are commonly treated as
    indicating non-conformity with Benford's Law.
    """
    deviations = [abs(observed.get(d, 0.0) - BENFORD_EXPECTED[d]) for d in DIGITS]
    return float(np.mean(deviations))


def compute_benford_metrics(amounts: list[float]) -> dict:
    """Compute the full set of Benford metrics for a list of transaction amounts.

    Returns a dict with `chi_square`, `mad`, `z_scores` (per digit), the
    `observed_distribution`, and `sample_size`.
    """
    observed = digit_distribution(amounts)
    n = sum(1 for a in amounts if first_digit(a) is not None)

    return {
        "chi_square": chi_square_statistic(observed, n),
        "mad": mean_absolute_deviation(observed),
        "z_scores": z_scores(observed, n),
        "observed_distribution": observed,
        "sample_size": n,
    }


def is_anomalous(metrics: dict, mad_threshold: float = 0.015) -> bool:
    """Whether a `compute_benford_metrics` result exceeds the MAD threshold."""
    return metrics["mad"] > mad_threshold


# ---------------------------------------------------------------------------
# Multivariate (cross-pair) Benford analysis
#
# A syndicate that splits wash volume evenly across N pairs keeps each pair's
# marginal digit distribution near Benford, so the univariate MAD/chi-square
# tests above see nothing. The coordination only shows up in the *joint*
# distribution: the pairs deviate from Benford in the same way at the same time.
# ---------------------------------------------------------------------------

_EXPECTED_VECTOR = np.array([BENFORD_EXPECTED[d] for d in DIGITS])


def _pair_series(trades: pd.DataFrame) -> pd.Series:
    """Return a per-row asset-pair label for `trades`.

    Uses an explicit `asset_pair` column when present, otherwise derives the
    pair from the `base_asset`/`counter_asset` dict columns.
    """
    if "asset_pair" in trades.columns:
        return trades["asset_pair"]

    def _symbol(asset: dict) -> str:
        code = asset["code"]
        issuer = asset.get("issuer")
        return code if issuer is None else f"{code}:{issuer}"

    return trades.apply(
        lambda r: f"{_symbol(r['base_asset'])}/{_symbol(r['counter_asset'])}", axis=1
    )


def joint_digit_matrix(
    trades: pd.DataFrame,
    pairs: list[str],
    window: pd.Timedelta | None = None,
) -> np.ndarray:
    """Build the joint leading-digit frequency matrix across `pairs`.

    Returns an array of shape ``(K, 9)`` where ``K = len(pairs)`` and row ``k``
    is the observed leading-digit frequency vector (digits 1-9) of pair ``k``'s
    `base_amount`s. When `window` is given and `trades` carries a
    `ledger_close_time` column, only trades within `window` of the most recent
    trade are used. Pairs with no trades contribute an all-zero row.
    """
    if trades is None or trades.empty:
        return np.zeros((len(pairs), 9))

    df = trades
    if window is not None and "ledger_close_time" in df.columns:
        times = pd.to_datetime(df["ledger_close_time"])
        cutoff = times.max() - window
        df = df.loc[times > cutoff]

    pair_labels = _pair_series(df)
    matrix = np.zeros((len(pairs), 9))
    for k, pair in enumerate(pairs):
        amounts = df.loc[pair_labels == pair, "base_amount"].tolist()
        dist = digit_distribution(amounts)
        matrix[k] = [dist[d] for d in DIGITS]
    return matrix


def _normal_scores(row: np.ndarray) -> np.ndarray:
    """Van der Waerden normal-score (Gaussian copula) transform of a vector."""
    order = row.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(row) + 1)
    return norm.ppf(ranks / (len(row) + 1))


def benford_copula_statistic(digit_matrix: np.ndarray) -> tuple[float, float]:
    """Test for coordinated cross-pair digit manipulation via a Gaussian copula.

    Each pair's deviation-from-Benford vector is mapped to Gaussian-copula
    pseudo-observations (normal scores), and the cross-pair correlation matrix is
    formed treating each pair as a variable observed over the 9 digits. Under the
    null — rows are i.i.d. Benford draws with zero copula correlation — the
    scaled sum of squared off-diagonal correlations is ``chi2`` distributed with
    ``C(K, 2)`` degrees of freedom. Coordinated pairs deviate from Benford in the
    same digit pattern, inflating the correlations and the statistic.

    Returns ``(statistic, p_value)``. A small p-value => coordinated manipulation.
    """
    matrix = np.asarray(digit_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0, 1.0

    deviations = matrix - _EXPECTED_VECTOR
    scored = np.vstack([_normal_scores(row) for row in deviations])

    corr = np.corrcoef(scored)
    corr = np.nan_to_num(corr, nan=0.0)

    k = matrix.shape[0]
    dof_per_corr = matrix.shape[1] - 1  # 9 digits, 1 lost to the copula transform
    upper = corr[np.triu_indices(k, k=1)]
    statistic = float(dof_per_corr * np.sum(upper**2))
    df = len(upper)
    p_value = float(chi2.sf(statistic, df)) if df > 0 else 1.0
    return statistic, p_value


def cross_pair_sync_score(
    trades: pd.DataFrame,
    pairs: list[str],
    window: pd.Timedelta = pd.Timedelta(minutes=1),
    z_threshold: float = 2.5,
    min_pairs: int = 3,
) -> float:
    """Fraction of time windows with simultaneous cross-pair digit anomalies.

    Buckets `trades` into `window`-sized bins; within each active bin a pair is
    "anomalous" when its maximum per-digit Benford Z-score exceeds `z_threshold`.
    A bin is *synchronised* when at least `min_pairs` pairs are simultaneously
    anomalous. Returns the fraction of active bins that are synchronised — high
    values indicate the pairs are being manipulated in concert.
    """
    if trades is None or trades.empty or "ledger_close_time" not in trades.columns:
        return 0.0

    df = trades.copy()
    df["ledger_close_time"] = pd.to_datetime(df["ledger_close_time"])
    df = df.assign(_pair=_pair_series(df).to_numpy())
    df = df[df["_pair"].isin(pairs)]
    if df.empty:
        return 0.0

    df = df.assign(
        _digit=df["base_amount"].map(first_digit),
        _bucket=df["ledger_close_time"].dt.floor(window),
    ).dropna(subset=["_digit"])
    if df.empty:
        return 0.0
    df["_digit"] = df["_digit"].astype(int)

    # Counts per (bucket, pair, digit) -> a (groups x 9) matrix, then a fully
    # vectorised per-group Benford Z-score (matching `z_scores`).
    counts_df = (
        df.groupby(["_bucket", "_pair", "_digit"]).size().unstack("_digit", fill_value=0)
    )
    counts_df = counts_df.reindex(columns=DIGITS, fill_value=0)
    counts = counts_df.to_numpy(dtype=float)
    n = counts.sum(axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.divide(counts, n, out=np.zeros_like(counts), where=n > 0)
        numerator = np.clip(np.abs(observed - _EXPECTED_VECTOR) - 1.0 / (2.0 * n), 0.0, None)
        denominator = np.sqrt(_EXPECTED_VECTOR * (1.0 - _EXPECTED_VECTOR) / n)
        z = np.where(denominator > 0, numerator / denominator, 0.0)
    max_z = z.max(axis=1)

    anomalous_per_bucket = (
        pd.Series(max_z > z_threshold, index=counts_df.index.get_level_values("_bucket"))
        .groupby(level=0)
        .sum()
    )
    active_bins = len(anomalous_per_bucket)
    sync_bins = int((anomalous_per_bucket >= min_pairs).sum())
    return float(sync_bins / active_bins) if active_bins else 0.0


def digit_entropy_delta(digit_matrix: np.ndarray) -> float:
    """Observed-minus-expected leading-digit entropy of the pooled distribution.

    The rows of `digit_matrix` are averaged into a single joint digit
    distribution whose Shannon entropy (nats) is compared to Benford's. A
    negative delta means the joint distribution is more concentrated than
    Benford predicts — the hallmark of coordinated round-number wash volume.
    """
    matrix = np.asarray(digit_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return 0.0

    pooled = matrix.mean(axis=0)
    total = pooled.sum()
    if total <= 0:
        return 0.0
    pooled = pooled / total

    observed_entropy = float(-sum(p * math.log(p) for p in pooled if p > 0))
    return observed_entropy - BENFORD_ENTROPY


def multivariate_benford_score(
    trades: pd.DataFrame,
    wallet_pairs: list[tuple[str, str]],
    window: pd.Timedelta = pd.Timedelta(hours=24),
) -> dict:
    """Multivariate Benford entry point for a set of `(wallet, pair)` combinations.

    Restricts `trades` to rows where one of the listed wallets traded one of the
    listed pairs, then computes the cross-pair copula statistic, the synchrony
    ratio, and the joint digit-entropy delta. Returns a dict with
    `copula_statistic`, `copula_pval`, `sync_ratio`, `digit_entropy_delta`, and
    the active `pairs`.
    """
    pairs = sorted({p for _, p in wallet_pairs})
    zero = {
        "copula_statistic": 0.0,
        "copula_pval": 1.0,
        "sync_ratio": 0.0,
        "digit_entropy_delta": 0.0,
        "pairs": pairs,
    }
    if trades is None or trades.empty or len(pairs) < 2:
        return zero

    wallets = {w for w, _ in wallet_pairs}
    df = trades
    base = df["base_account"] if "base_account" in df.columns else pd.Series(index=df.index, dtype=object)
    counter = df["counter_account"] if "counter_account" in df.columns else pd.Series(index=df.index, dtype=object)
    df = df[base.isin(wallets) | counter.isin(wallets)]
    if df.empty:
        return zero

    matrix = joint_digit_matrix(df, pairs, window)
    statistic, pval = benford_copula_statistic(matrix)
    return {
        "copula_statistic": statistic,
        "copula_pval": pval,
        "sync_ratio": cross_pair_sync_score(df, pairs),
        "digit_entropy_delta": digit_entropy_delta(matrix),
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# Adaptive window sizing
#
# The chi-square test over 9 digit bins requires expected cell counts >= 5,
# which demands N >= ~30 trades. For sparse wallets or quiet periods the fixed
# rolling windows (1h, 4h, 24h, 7d, 30d) may have too few samples.
# AdaptiveBenfordWindow doubles the window until N >= MIN_SAMPLE_COUNT or the
# maximum window width is reached, keeping statistics valid.
# ---------------------------------------------------------------------------


@dataclass
class BenfordWindowResult:
    """Result of an adaptive Benford window computation for one target window.

    Attributes:
        label: The target window label (e.g. ``"1h"``).
        amounts: Trade amounts sliced from the (possibly expanded) window.
        effective_seconds: Actual window width used (>= target when expanded).
        valid: ``True`` when ``len(amounts) >= min_sample_count``.
        expanded: ``True`` when the window was widened beyond the target.
        merged: ``True`` when two adjacent windows were merged as a fallback.
        merged_windows: Labels of the constituent windows when ``merged=True``.
        reason: Human-readable explanation when ``valid=False``.
    """

    label: str
    amounts: list[float]
    effective_seconds: int
    valid: bool
    expanded: bool = False
    merged: bool = False
    merged_windows: list[str] = field(default_factory=list)
    reason: str = ""


class AdaptiveBenfordWindow:
    """Adaptive rolling-window sampler for Benford's Law analysis.

    Guarantees that every returned ``BenfordWindowResult`` either contains at
    least ``min_sample_count`` trades (``valid=True``) or is explicitly marked
    ``valid=False`` — never silently returning unreliable statistics.

    Algorithm (per target window):
    1. Try the target window as-is.
    2. If N < ``min_sample_count``, double the window up to ``max_window_days``.
    3. If still N < ``min_sample_count``, return ``valid=False``.
    4. As a post-pass, merge the two smallest invalid windows if that reaches N.

    Trade timestamps are pre-sorted once; individual window slices use
    ``bisect`` for O(log N) boundary lookups.

    Args:
        min_sample_count: Minimum trades for a statistically valid chi-square.
        max_window_days: Hard upper bound on expansion (security: prevents
            loading unbounded history and exhausting memory).
        expansion_factor: Multiplier applied per expansion step (default 2.0).
    """

    def __init__(
        self,
        min_sample_count: int = 30,
        max_window_days: int = 90,
        expansion_factor: float = 2.0,
    ) -> None:
        if max_window_days > 365:
            raise ValueError("max_window_days must not exceed 365 to prevent memory exhaustion")
        self.min_sample_count = min_sample_count
        self.max_window_days = max_window_days
        self.expansion_factor = expansion_factor

    @staticmethod
    def _prepare_sorted_arrays(trades: list[dict]) -> tuple[list[float], list[float]]:
        """Sort trades by timestamp and pre-filter to only valid amounts.

        Returns two parallel lists: sorted timestamps and corresponding
        positive-finite amounts.  Pre-filtering here means per-window slices
        need no further validation, reducing inner-loop cost from O(slice)
        to O(1).
        """
        valid_pairs = sorted(
            (t["timestamp"], t["amount"])
            for t in trades
            if t.get("amount") is not None
            and math.isfinite(t["amount"])
            and t["amount"] > 0
        )
        sorted_timestamps_arr = [p[0] for p in valid_pairs]
        sorted_amounts_arr = [p[1] for p in valid_pairs]
        return sorted_timestamps_arr, sorted_amounts_arr

    def fit(
        self,
        trades: list[dict],
        target_window_label: str,
        target_window_seconds: int,
        as_of_ts: float,
    ) -> BenfordWindowResult:
        """Compute a Benford window result for one target window.

        Args:
            trades: List of trade dicts with ``timestamp`` (Unix epoch float)
                and ``amount`` (positive float) keys.
            target_window_label: Human-readable label (e.g. ``"1h"``).
            target_window_seconds: Width of the target window in seconds.
            as_of_ts: Reference Unix epoch timestamp (right edge of window).

        Returns:
            ``BenfordWindowResult`` with amounts and validity flag.
        """
        if not trades:
            return BenfordWindowResult(
                label=target_window_label,
                amounts=[],
                effective_seconds=target_window_seconds,
                valid=False,
                reason="no_trades",
            )

        sorted_timestamps_arr, sorted_amounts_arr = self._prepare_sorted_arrays(trades)
        return self._fit_presorted(
            sorted_timestamps_arr,
            sorted_amounts_arr,
            target_window_label,
            target_window_seconds,
            as_of_ts,
        )

    def _fit_presorted(
        self,
        sorted_timestamps_arr: list[float],
        sorted_amounts_arr: list[float],
        target_window_label: str,
        target_window_seconds: int,
        as_of_ts: float,
    ) -> BenfordWindowResult:
        """Inner loop operating on pre-sorted, pre-filtered parallel arrays.

        Called by both ``fit`` (which sorts and filters on entry) and
        ``fit_all`` (which sorts and filters once for all windows).

        Callers must guarantee:
        - ``sorted_timestamps_arr`` and ``sorted_amounts_arr`` are sorted by
          timestamp ascending and have the same length.
        - All amounts are positive and finite (pre-filtered by
          ``_prepare_sorted_arrays``).
        """
        max_seconds = int(self.max_window_days * 86400)
        max_iterations = (
            int(math.ceil(math.log2(max_seconds / max(target_window_seconds, 1)))) + 1
            if max_seconds > target_window_seconds
            else 1
        )

        width = target_window_seconds
        expanded = False
        valid_amounts: list[float] = []

        for _ in range(max_iterations + 1):
            width = min(width, max_seconds)
            cutoff = as_of_ts - width
            left_idx = bisect.bisect_right(sorted_timestamps_arr, cutoff)
            # All amounts are pre-filtered; the slice is the valid set.
            valid_amounts = sorted_amounts_arr[left_idx:]
            if len(valid_amounts) >= self.min_sample_count:
                if width == target_window_seconds:
                    logger.debug(
                        "Benford window %s: N=%d >= %d (no expansion needed)",
                        target_window_label,
                        len(valid_amounts),
                        self.min_sample_count,
                    )
                else:
                    orig_idx = bisect.bisect_right(
                        sorted_timestamps_arr, as_of_ts - target_window_seconds
                    )
                    orig_n = len(sorted_amounts_arr) - orig_idx
                    logger.warning(
                        "Benford window %s expanded: original_N=%d -> final_N=%d, "
                        "effective_width_hours=%.1f",
                        target_window_label,
                        orig_n,
                        len(valid_amounts),
                        width / 3600,
                    )
                return BenfordWindowResult(
                    label=target_window_label,
                    amounts=list(valid_amounts),
                    effective_seconds=width,
                    valid=True,
                    expanded=expanded,
                )
            if width >= max_seconds:
                break
            width = int(min(width * self.expansion_factor, max_seconds))
            expanded = True

        logger.error(
            "Benford window %s: insufficient data even at max_width=%dd (N=%d < %d)",
            target_window_label,
            self.max_window_days,
            len(valid_amounts),
            self.min_sample_count,
        )
        return BenfordWindowResult(
            label=target_window_label,
            amounts=list(valid_amounts),
            effective_seconds=width,
            valid=False,
            reason="insufficient_even_after_expansion",
            expanded=expanded,
        )

    def fit_all(
        self,
        trades: list[dict],
        windows: dict[str, int],
        as_of_ts: float,
    ) -> dict[str, BenfordWindowResult]:
        """Compute adaptive Benford windows for all target windows.

        Trades are sorted and filtered once; each window uses ``bisect`` for
        O(log N) boundary lookups, giving O(N log N + W log N) total
        complexity where W is the number of windows.

        Args:
            trades: List of trade dicts (``timestamp``, ``amount``).
            windows: Mapping of label -> seconds for each target window.
            as_of_ts: Right-edge timestamp.

        Returns:
            Dict mapping window label -> ``BenfordWindowResult``.

        After individual fits, attempts one merge pass: if two of the smallest
        invalid windows can be combined to reach ``min_sample_count``, they are
        merged and marked ``merged=True``.
        """
        if not trades:
            return {
                label: BenfordWindowResult(
                    label=label,
                    amounts=[],
                    effective_seconds=secs,
                    valid=False,
                    reason="no_trades",
                )
                for label, secs in windows.items()
            }

        # Sort and filter once; all window calls reuse the pre-filtered arrays.
        sorted_timestamps_arr, sorted_amounts_arr = self._prepare_sorted_arrays(trades)

        results = {
            label: self._fit_presorted(
                sorted_timestamps_arr, sorted_amounts_arr, label, secs, as_of_ts
            )
            for label, secs in windows.items()
        }

        # Merge pass: find smallest two invalid windows
        invalid = [(label, r) for label, r in results.items() if not r.valid]
        if len(invalid) >= 2:
            invalid_sorted = sorted(invalid, key=lambda x: x[1].effective_seconds)
            label_a, res_a = invalid_sorted[0]
            label_b, res_b = invalid_sorted[1]
            merged_amounts = res_a.amounts + res_b.amounts
            if len(merged_amounts) >= self.min_sample_count:
                merged = BenfordWindowResult(
                    label=label_a,
                    amounts=merged_amounts,
                    effective_seconds=max(res_a.effective_seconds, res_b.effective_seconds),
                    valid=True,
                    merged=True,
                    merged_windows=[label_a, label_b],
                )
                results[label_a] = merged
        return results
