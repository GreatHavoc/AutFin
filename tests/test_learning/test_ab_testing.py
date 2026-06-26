"""Tests for A/B testing framework."""

import pytest
from src.autfin.learning.ab_testing import (
    ABTestingFramework,
    Experiment,
    ExperimentResult,
    get_ab_framework,
    reset_ab_framework,
)


class TestExperiment:
    def test_create_experiment(self):
        fw = ABTestingFramework()
        e = fw.create_experiment("test", "control", "treatment")
        assert e.name == "test"
        assert e.variant_a == "control"
        assert e.variant_b == "treatment"
        assert e.is_active is True

    def test_create_with_custom_id(self):
        fw = ABTestingFramework()
        e = fw.create_experiment("test", "A", "B", experiment_id="my-exp")
        assert e.experiment_id == "my-exp"

    def test_conversion_property(self):
        e = Experiment(
            experiment_id="x", name="t", variant_a="A", variant_b="B",
            sample_size_a=10, outcome_sum_a=5.0,
            sample_size_b=10, outcome_sum_b=7.0,
        )
        assert e.conversion_a == 0.5
        assert e.conversion_b == 0.7


class TestVariantAssignment:
    def test_assignment_is_deterministic(self):
        fw = ABTestingFramework()
        e = fw.create_experiment("x", "A", "B", experiment_id="exp1")
        v1 = fw.assign_variant("exp1", "RELIANCE")
        v2 = fw.assign_variant("exp1", "RELIANCE")
        assert v1 == v2

    def test_different_subjects_can_get_different_variants(self):
        fw = ABTestingFramework()
        fw.create_experiment("y", "A", "B", experiment_id="exp2")
        # Get variants for many symbols — should be roughly 50/50
        variants = [fw.assign_variant("exp2", f"SYM{i:03d}") for i in range(100)]
        a_count = variants.count("A")
        b_count = variants.count("B")
        assert a_count > 30 and a_count < 70
        assert b_count > 30 and b_count < 70

    def test_unknown_experiment_returns_default(self):
        fw = ABTestingFramework()
        v = fw.assign_variant("nonexistent", "X")
        assert v == "A"


class TestOutcomeRecording:
    def test_record_outcome_increments_correct_variant(self):
        fw = ABTestingFramework()
        e = fw.create_experiment("z", "A", "B", experiment_id="exp3")
        variant = fw.assign_variant("exp3", "STOCK1")
        fw.record_outcome("exp3", "STOCK1", outcome=0.05)
        if variant == "A":
            assert e.sample_size_a == 1
            assert e.outcome_sum_a == 0.05
        else:
            assert e.sample_size_b == 1
            assert e.outcome_sum_b == 0.05


class TestComputeResults:
    def test_inconclusive_with_insufficient_samples(self):
        fw = ABTestingFramework()
        e = fw.create_experiment("z", "A", "B", experiment_id="exp4")
        variant = fw.assign_variant("exp4", "S1")
        fw.record_outcome("exp4", "S1", outcome=0.01)
        result = fw.compute_results("exp4")
        assert result is not None
        assert result.winner == "inconclusive"
        assert result.sample_size < 20

    def test_winner_selected(self):
        fw = ABTestingFramework()
        e = fw.create_experiment("z", "A", "B", experiment_id="exp5")
        # Record many wins for A — but assign_variant will distribute
        # Use direct recording - we need to track by variant
        for i in range(50):
            variant = fw.assign_variant("exp5", f"SYM_{i:03d}")
            fw.record_outcome("exp5", f"SYM_{i:03d}", outcome=1.0 if variant == "A" else 0.3)
        result = fw.compute_results("exp5")
        assert result is not None
        assert result.sample_size == 50

    def test_results_for_unknown_experiment_returns_none(self):
        fw = ABTestingFramework()
        result = fw.compute_results("nonexistent")
        assert result is None


class TestExperimentResult:
    def test_result_fields(self):
        r = ExperimentResult(
            experiment_id="x",
            winner="A",
            confidence=0.85,
            lift=20.0,
            sample_size=50,
            sample_size_a=25,
            sample_size_b=25,
            recommendation="A wins",
        )
        assert r.winner == "A"
        assert r.confidence == 0.85
        assert r.lift == 20.0


class TestSingleton:
    def test_get_ab_framework_singleton(self):
        reset_ab_framework()
        f1 = get_ab_framework()
        f2 = get_ab_framework()
        assert f1 is f2

    def test_reset_ab_framework(self):
        f1 = get_ab_framework()
        reset_ab_framework()
        f2 = get_ab_framework()
        assert f1 is not f2


class TestListExperiments:
    def test_list_returns_all_experiments(self):
        fw = ABTestingFramework()
        fw.create_experiment("e1", "A", "B")
        fw.create_experiment("e2", "C", "D")
        exps = fw.list_experiments()
        assert len(exps) == 2
