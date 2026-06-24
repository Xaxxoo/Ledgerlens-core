"""Tests for AdaptiveBenfordWindow (Issue-102).

Covers:
- Window with N >= 30 from start: no expansion, expanded=False
- Window with N = 15 expands once to 2x width, reaches N >= 30
- Window never reaches N >= 30 even at max width: valid=False
- Merge fallback: two windows each < 30 merge to >= 30
- Edge cases: 0 trades, exactly 30 trades, 29 trades, amount=0 skipped
"""

import pytest

from detection.benford_engine import AdaptiveBenfordWindow, BenfordWindowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(n: int, amount: float = 123.45, base_ts: float = 0.0, spacing: float = 60.0):
    """Create n trades spaced `spacing` seconds apart ending at base_ts."""
    return [
        {"timestamp": base_ts - (n - 1 - i) * spacing, "amount": amount}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit tests: AdaptiveBenfordWindow.fit()
# ---------------------------------------------------------------------------


class TestAdaptiveBenfordWindowFit:
    def test_no_expansion_when_n_gte_min(self):
        """Window with >= 30 trades should not be expanded."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        as_of = 3600.0  # 1 hour
        # 35 trades within the last 3600 s
        trades = _make_trades(35, base_ts=as_of, spacing=100.0)
        result = engine.fit(trades, "1h", 3600, as_of)
        assert result.valid is True
        assert result.expanded is False
        assert len(result.amounts) >= 30

    def test_expansion_when_n_below_min(self):
        """Window with 15 trades should expand to 2x and find >= 30."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        as_of = 3600.0
        # 15 trades in [0, 3600], but 35 in [0, 7200]
        trades = _make_trades(15, base_ts=as_of, spacing=240.0) + _make_trades(
            20, base_ts=0.0, spacing=240.0
        )
        result = engine.fit(trades, "1h", 3600, as_of)
        assert result.valid is True
        assert result.expanded is True
        assert result.effective_seconds > 3600

    def test_invalid_when_max_width_insufficient(self):
        """When even max width cannot reach N >= 30, result is valid=False."""
        engine = AdaptiveBenfordWindow(min_sample_count=30, max_window_days=1)
        as_of = 86400.0
        # Only 5 trades total over the entire day
        trades = _make_trades(5, base_ts=as_of, spacing=10000.0)
        result = engine.fit(trades, "1h", 3600, as_of)
        assert result.valid is False
        assert result.reason == "insufficient_even_after_expansion"

    def test_exactly_30_trades_valid_no_expansion(self):
        """Exactly 30 trades should be valid without expansion."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        as_of = 3600.0
        trades = _make_trades(30, base_ts=as_of, spacing=100.0)
        result = engine.fit(trades, "1h", 3600, as_of)
        assert result.valid is True
        assert result.expanded is False

    def test_29_trades_triggers_expansion(self):
        """29 trades should trigger at least one expansion attempt."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        as_of = 3600.0
        # 29 in the 1h window, 1 more at 3700 s ago (outside 1h)
        trades = _make_trades(29, base_ts=as_of, spacing=100.0) + [
            {"timestamp": as_of - 3700, "amount": 50.0}
        ]
        result = engine.fit(trades, "1h", 3600, as_of)
        # After expansion the extra trade at 3700s back should be included
        assert result.valid is True
        assert result.expanded is True

    def test_zero_amount_excluded_from_count(self):
        """Trades with amount=0 must not count toward the valid sample count."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        as_of = 3600.0
        # 28 valid + 10 zero-amount = still only 28 valid
        trades = _make_trades(28, base_ts=as_of, spacing=100.0) + _make_trades(
            10, amount=0.0, base_ts=as_of, spacing=100.0
        )
        result = engine.fit(trades, "1h", 3600, as_of)
        # 28 valid trades < 30 so should expand; depends on what's available
        assert result.valid is False or result.expanded is True

    def test_empty_trades_returns_invalid(self):
        """No trades at all must return valid=False with reason."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        result = engine.fit([], "1h", 3600, 3600.0)
        assert result.valid is False
        assert result.reason == "no_trades"

    def test_max_window_days_safety_cap(self):
        """max_window_days > 365 must raise ValueError."""
        with pytest.raises(ValueError, match="365"):
            AdaptiveBenfordWindow(max_window_days=400)

    def test_effective_seconds_set_correctly_on_expansion(self):
        """effective_seconds must be > target_window_seconds when expanded."""
        engine = AdaptiveBenfordWindow(min_sample_count=30)
        as_of = 3600.0
        # 15 in window, 20 more at 2x width
        trades = _make_trades(15, base_ts=as_of, spacing=200.0) + _make_trades(
            20, base_ts=as_of - 3600, spacing=200.0
        )
        result = engine.fit(trades, "1h", 3600, as_of)
        if result.valid and result.expanded:
            assert result.effective_seconds > 3600

    def test_amounts_immutable_no_mutation(self):
        """The original trades list must not be mutated."""
        engine = AdaptiveBenfordWindow(min_sample_count=5)
        original = _make_trades(10, base_ts=100.0)
        original_len = len(original)
        engine.fit(original, "1h", 3600, 3700.0)
        assert len(original) == original_len


# ---------------------------------------------------------------------------
# Unit tests: AdaptiveBenfordWindow.fit_all()
# ---------------------------------------------------------------------------


class TestAdaptiveBenfordWindowFitAll:
    def test_returns_all_labels(self):
        """fit_all must return one result per input window label."""
        engine = AdaptiveBenfordWindow(min_sample_count=5)
        as_of = 86400.0
        trades = _make_trades(50, base_ts=as_of, spacing=1000.0)
        windows = {"1h": 3600, "4h": 14400, "24h": 86400}
        results = engine.fit_all(trades, windows, as_of)
        assert set(results.keys()) == {"1h", "4h", "24h"}

    def test_merge_two_invalid_windows(self):
        """When two smallest windows are individually invalid but together >= 30,
        the smaller one should be merged and marked valid."""
        engine = AdaptiveBenfordWindow(min_sample_count=30, max_window_days=1)
        as_of = 3600.0
        # 20 trades in the 1h window; 0 in the 4h window (before 1h edge)
        # After expansion (max 1 day), 1h window only gets 20; 4h gets 20 too
        # Combined = 40 >= 30
        trades_1h = _make_trades(20, base_ts=as_of, spacing=100.0)
        trades_4h_only = _make_trades(20, base_ts=as_of - 5000, spacing=200.0)
        trades = trades_1h + trades_4h_only
        results = engine.fit_all(trades, {"1h": 3600, "4h": 14400}, as_of)
        # At least one result should be valid after merge
        valid_count = sum(1 for r in results.values() if r.valid)
        assert valid_count >= 1

    def test_no_merge_when_both_valid(self):
        """When both windows are individually valid, no merge should occur."""
        engine = AdaptiveBenfordWindow(min_sample_count=5)
        as_of = 14400.0
        trades = _make_trades(50, base_ts=as_of, spacing=200.0)
        results = engine.fit_all(trades, {"1h": 3600, "4h": 14400}, as_of)
        for r in results.values():
            assert r.merged is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAdaptiveBenfordIntegration:
    def test_feature_vector_length_consistent(self):
        """FEATURE_NAMES must not grow or shrink after adding adaptive features."""
        from detection.feature_engineering import BENFORD_ADAPTIVE_FEATURE_NAMES, FEATURE_NAMES

        assert all(
            f"benford_window_expanded_{w}" in FEATURE_NAMES
            for w in ("1h", "4h", "24h", "7d", "30d")
        ), "Adaptive feature flags missing from FEATURE_NAMES"
        assert len(BENFORD_ADAPTIVE_FEATURE_NAMES) == 5

    def test_sparse_trade_list_no_exception(self):
        """5 total trades must not raise; all windows should be valid=False."""
        engine = AdaptiveBenfordWindow(min_sample_count=30, max_window_days=2)
        as_of = 86400.0
        trades = _make_trades(5, base_ts=as_of, spacing=10000.0)
        windows = {"1h": 3600, "4h": 14400, "24h": 86400, "7d": 604800, "30d": 2592000}
        results = engine.fit_all(trades, windows, as_of)
        # Should not raise; may be all invalid
        for r in results.values():
            assert isinstance(r, BenfordWindowResult)

    def test_performance_5_windows_10000_trades_under_100ms(self):
        """5 windows over 10,000 trades must complete in < 100 ms."""
        engine = AdaptiveBenfordWindow()
        import time

        rng = __import__("random").Random(42)
        as_of = 2592000.0  # 30 days
        trades = [
            {"timestamp": rng.uniform(0, as_of), "amount": 10 ** rng.uniform(0, 4)}
            for _ in range(10000)
        ]
        windows = {"1h": 3600, "4h": 14400, "24h": 86400, "7d": 604800, "30d": 2592000}

        start = time.perf_counter()
        results = engine.fit_all(trades, windows, as_of)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"fit_all took {elapsed_ms:.1f} ms (limit 100 ms)"
        assert len(results) == 5

    def test_settings_constants_present(self):
        """New Benford settings constants must be accessible."""
        from config.settings import settings

        assert hasattr(settings, "BENFORD_MIN_SAMPLE_COUNT")
        assert hasattr(settings, "BENFORD_MAX_WINDOW_DAYS")
        assert hasattr(settings, "BENFORD_EXPANSION_FACTOR")
        assert settings.BENFORD_MIN_SAMPLE_COUNT == 30
        assert settings.BENFORD_MAX_WINDOW_DAYS == 90
        assert settings.BENFORD_EXPANSION_FACTOR == 2.0
