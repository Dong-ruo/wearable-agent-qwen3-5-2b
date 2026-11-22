import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lightagent.cases import REQUEST, cases
from lightagent.environment import Environment
from lightagent.evaluation import assess
from lightagent.models import Call
from lightagent.policy import EndpointPolicy, RulePolicy
from lightagent.runtime import Runtime


@pytest.fixture
def endpoint():
    reference = RulePolicy()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            assert self.path == "/v1/chat/completions"
            action = reference.next(payload["messages"], payload["tools"])
            if isinstance(action, Call):
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "remote-call", "type": "function",
                           "function": {"name": action.name, "arguments": json.dumps(action.arguments, ensure_ascii=False)}}]}
            else:
                message = {"role": "assistant", "content": "<think>reference test only</think>" + action.model_dump_json()}
            body = json.dumps({"choices": [{"message": message}]}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def test_full_tool_loop_over_compatible_http_transport(endpoint):
    url, requests = endpoint
    case = cases()["sunny"]
    runtime = Runtime(Environment(case.scenario), EndpointPolicy(url, "test-model", timeout=3))
    assert runtime.run(REQUEST).status == "success"
    assert assess(case, runtime)["passed"]
    assert len(requests) == 9
    assert all(r["model"] == "test-model" and r["parallel_tool_calls"] is False for r in requests)
    assert len(requests[-1]["tools"]) == 14
    assert requests[-1]["messages"][-1]["role"] == "tool"


def test_unreachable_endpoint_reports_failure_without_writes():
    runtime = Runtime(Environment(cases()["sunny"].scenario), EndpointPolicy("http://127.0.0.1:1/v1", "unused", timeout=1))
    final = runtime.run(REQUEST)
    assert final.status == "failed"
    assert not runtime.env.events and not runtime.env.reminders


def test_bad_endpoint_url_is_rejected():
    with pytest.raises(ValueError):
        EndpointPolicy("file:///private", "unused")
