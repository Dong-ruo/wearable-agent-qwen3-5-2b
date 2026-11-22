from wearable_tasks.contract import digest
from wearable_tasks.templates import build_case, shape_for


def case(kind="create", ordinal=0, variant=0, split="train"):
    semantic = {"kind": kind, "world_shape": shape_for(ordinal)}
    group = {"semantic_spec": semantic, "group_id": digest(semantic), "seed": 42,
             "calendar_index": ordinal, "split": split, "robust": split == "test_robustness"}
    return build_case(group, variant)


def empty_case():
    task = case()
    task["initial_state"]["events"] = []
    task["initial_state"]["reminders"] = []
    return task
