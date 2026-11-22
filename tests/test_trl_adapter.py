import inspect
import json

from lightagent.cases import REQUEST, scenario
from lightagent.environment import SPECS
from lightagent.integrations.trl_env import make_environment_factory


def test_factory_reset_and_public_tools_contract():
    factory = make_environment_factory(scenario(), REQUEST)
    assert not inspect.signature(factory).parameters
    a, b = factory(), factory()
    assert REQUEST in a.reset()
    public = {name for name, member in inspect.getmembers(a, callable) if not name.startswith("_")}
    assert public == set(SPECS) | {"reset"}
    for name in public - {"reset"}:
        method = getattr(a, name)
        assert method.__doc__
        for parameter in inspect.signature(method).parameters.values():
            assert parameter.annotation is not inspect.Parameter.empty
            assert parameter.name + ":" in method.__doc__
    result = json.loads(a.calendar_create("户外跑步", "outdoor_run", "杭州", "2026-11-17T18:00:00+08:00", 30, "task-1"))
    assert result["ok"] and len(a._env.events) == 1
    assert not b._env.events
    a.reset()
    assert not a._env.events
    assert json.loads(a.clock_now())["data"]["timezone"] == "Asia/Shanghai"
