from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from dspy_avqa import latency


def test_retry_adjusted_latency_excludes_failed_attempt_and_backoff(monkeypatch):
    times = iter([0.0, 2.0, 5.0, 9.0, 10.0])
    monkeypatch.setattr(latency.time, "perf_counter", lambda: next(times))
    tracker = latency.LatencyTracker()

    tracker.start_attempt("planner")
    tracker.finish_attempt(success=False)
    tracker.start_attempt("planner")
    tracker.finish_attempt(success=True)

    assert tracker.summary(complete=True) == {
        "wall_clock_seconds": 10.0,
        "retry_adjusted_seconds": 5.0,
        "successful_model_call_seconds": 4.0,
        "planner_successful_calls": 1,
        "perception_successful_calls": 0,
    }


def test_incomplete_latency_has_null_adjusted_time(monkeypatch):
    times = iter([4.0, 7.0, 8.0])
    monkeypatch.setattr(latency.time, "perf_counter", lambda: next(times))
    tracker = latency.LatencyTracker()

    tracker.start_attempt("perception")
    tracker.finish_attempt(success=False)

    assert tracker.summary(complete=False) == {
        "wall_clock_seconds": 4.0,
        "retry_adjusted_seconds": None,
        "successful_model_call_seconds": 0.0,
        "planner_successful_calls": 0,
        "perception_successful_calls": 0,
    }


def test_latency_context_is_noop_without_active_sample():
    with latency.model_component("planner"):
        latency.start_model_attempt()
        latency.finish_model_attempt(success=True)


def test_latency_context_is_isolated_between_worker_threads():
    def run_sample(component: str):
        with latency.sample_latency() as tracker:
            with latency.model_component(component):
                latency.start_model_attempt()
                latency.finish_model_attempt(success=True)
        return tracker.summary(complete=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        planner_summary, perception_summary = executor.map(
            run_sample, ("planner", "perception")
        )

    assert planner_summary["planner_successful_calls"] == 1
    assert planner_summary["perception_successful_calls"] == 0
    assert perception_summary["planner_successful_calls"] == 0
    assert perception_summary["perception_successful_calls"] == 1
