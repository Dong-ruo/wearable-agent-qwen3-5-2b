"""Explicit staged unittest runner. Only invoked when the user executes the check command."""
from __future__ import annotations

import json
import platform
import time
import unittest

from .contract import schema_hash
from .pipeline import source_fingerprint


def run_checks(output):
    started = time.perf_counter()
    phases = ["wearable_tasks.tests.test_core", "wearable_tasks.tests.test_levels.L123Tests",
              "wearable_tasks.tests.test_levels.L456Tests", "wearable_tasks.tests.test_pipeline"]
    results, passed = [], True
    for name in phases:
        suite = unittest.defaultTestLoader.loadTestsFromName(name)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        results.append({"phase": name, "tests_run": result.testsRun, "failures": len(result.failures),
                        "errors": len(result.errors), "skipped": len(result.skipped),
                        "details": [{"test": str(test), "traceback": text} for test, text in result.failures+result.errors]})
        if not result.wasSuccessful():
            passed = False
            break
    report = {"status": "passed" if passed else "failed", "python_version": platform.python_version(),
              "source_fingerprint": source_fingerprint(), "schema_hash": schema_hash(),
              "elapsed_seconds": time.perf_counter()-started, "phases": results,
              "tests_run": sum(r["tests_run"] for r in results), "model_or_gpu_used": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
