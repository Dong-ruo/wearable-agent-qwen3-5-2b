import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lightagent.profiling import profile


def test_streaming_ttft_measures_first_content_not_role_only_event():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert payload["stream"] and payload["stream_options"]["include_usage"]
            events = [
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "结果"}}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}},
            ]
            body = ("".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = profile(f"http://127.0.0.1:{server.server_port}/v1", {"model": "test", "messages": [{"role": "user", "content": "测试"}]})
        assert result["generated_tokens"] == 2
        assert 0 < result["ttft_http_seconds"] <= result["model_http_seconds"]
        assert result["peak_model_memory_bytes"] is None
        assert not result["tool_execution_included"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
