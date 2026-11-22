import copy
import json
import tempfile
import unittest
from pathlib import Path

from wearable_tasks.auditing import audit_splits
from wearable_tasks.contract import digest, stamp
from wearable_tasks.exporting import ReferenceEnvironmentLoader, TaskStore, assert_public, sft_record
from wearable_tasks.pipeline import build, build_tasks, check_prerequisites, dataset_hash, load_configuration
from wearable_tasks.reference import run_reference
from wearable_tasks.templates import plan_groups


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_configuration(ROOT / "configs/smoke.json")
        cls.tasks = build_tasks(cls.config)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "smoke"
        cls.report = build(cls.config, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_smoke_has_six_levels_and_all_five_splits(self):
        self.assertEqual(len(self.tasks), 100)
        self.assertEqual({t["level"] for t in self.tasks}, set(range(1, 7)))
        self.assertEqual({t["split"] for t in self.tasks}, set(self.config["counts"]))
        self.assertEqual(self.report["reference_successes"], 100)
        self.assertEqual(self.report["status"], "passed")
        self.assertEqual(self.report["checks"]["isolation_cases"], 100)
        self.assertEqual(self.report["checks"]["replay_cases"], 100)
        self.assertEqual(self.report["checks"]["privacy_cases"], 100)
        dates = [(stamp(t["visible_context"]["now"]), stamp(t["reference_intent"]["date"]+"T00:00:00+08:00")) for t in self.tasks]
        self.assertTrue(any(a.month != b.month for a, b in dates))
        self.assertTrue(any(a.year != b.year for a, b in dates))

    def test_seed_rebuild_groups_and_planned_formal_counts(self):
        self.assertEqual(dataset_hash(self.tasks), dataset_hash(build_tasks(self.config)))
        self.assertTrue(audit_splits(self.tasks)["passed"])
        formal = load_configuration(ROOT / "configs/formal.json")
        groups = plan_groups(formal["counts"], formal["seed"])
        self.assertEqual(len(groups)*4, 2900)
        self.assertEqual(len({g["group_id"] for g in groups}), len(groups))
        self.assertEqual(formal["counts"], {"train": 2000, "val": 300, "test_id": 200, "test_composition": 200, "test_robustness": 200})

    def test_group_duplicate_and_composition_leaks_are_rejected(self):
        bad = copy.deepcopy(self.tasks)
        leaked = copy.deepcopy(bad[0])
        leaked.update(case_id="extra-id", split="val")
        bad.append(leaked)
        with self.assertRaises(ValueError):
            audit_splits(bad)
        bad = copy.deepcopy(self.tasks)
        leaked = copy.deepcopy(bad[0])
        leaked["semantic_spec"]["world_shape"]["spacing"] += 1
        leaked.update(case_id="another-id", split="val", group_id=digest(leaked["semantic_spec"]))
        bad.append(leaked)
        with self.assertRaises(ValueError):
            audit_splits(bad)
        bad = copy.deepcopy(self.tasks)
        for task in bad:
            if task["split"] == "test_composition":
                task["split"] = "train"
        with self.assertRaises(ValueError):
            audit_splits(bad)

    def test_all_applicable_negative_categories_are_batch_rejected(self):
        counts = self.report["negative_samples_rejected"]
        for category in ("wrong_activity", "wrong_time", "missing_reminder", "wrong_reminder_binding", "duplicate_creation",
                         "deleted_unrelated_event", "guessed_missing_slot", "posthoc_clarification", "ignored_followup",
                         "missing_necessary_query", "fabricated_query_result"):
            self.assertGreater(counts.get(category, 0), 0, category)
        self.assertGreater(self.report["negative_samples_rejected_total"], 100)

    def test_sft_is_public_success_only_and_test_demonstrations_not_exported(self):
        count = 0
        for split in ("train", "val"):
            path = self.output / "public/sft" / (split+".jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), self.config["counts"][split])
            for row in rows:
                self.assertEqual(set(row), {"messages", "tools"})
                assert_public(row)
                count += 1
        self.assertEqual(count, 52)
        self.assertFalse((self.output / "public/sft/test_id.jsonl").exists())
        task = next(t for t in self.tasks if t["level"] == 4)
        env = run_reference(task)
        record = sft_record(task, env.trajectory())
        self.assertTrue(any(m["role"] == "user" and "=" in m["content"] for m in record["messages"]))
        bad = env.trajectory()
        bad["messages"][-1]["content"] = '{"goal":{"secret":1}}'
        with self.assertRaises(ValueError):
            sft_record(task, bad)

    def test_private_loader_resets_and_public_references_do_not_inline_goals(self):
        task = self.tasks[0]
        stored = TaskStore(self.output).get(task["case_id"])
        self.assertEqual(stored, task)
        loader = ReferenceEnvironmentLoader(self.output)
        observation = loader.reset(task["case_id"])
        assert_public(observation)
        with self.assertRaises(ValueError):
            loader.terminal_reward()
        refs = self.output / "public/grpo/train.jsonl"
        for line in refs.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            self.assertIn("case_id", row)
            assert_public(row)

    def test_formal_is_gated_and_nonempty_output_is_not_silently_overwritten(self):
        formal = load_configuration(ROOT / "configs/formal.json")
        with self.assertRaises(ValueError):
            check_prerequisites(formal)
        with self.assertRaises(ValueError):
            build(self.config, self.output)
