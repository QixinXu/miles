"""Miles client and isolated ALE-Bench public judge service.

The judge executes candidate code through ALE-Bench/Docker. It binds only to
localhost by default and requires a shared bearer token on every request.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import threading
from collections import Counter
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from qwen_eval.frontiercs_ttt.types import JudgeFeedback


def _token() -> str:
    token = os.environ.get("ALE_BENCH_JUDGE_TOKEN", "")
    if len(token) < 24:
        raise RuntimeError("set ALE_BENCH_JUDGE_TOKEN to a random secret of at least 24 characters")
    return token


class AleBenchJudge:
    def __init__(self, *, judge_url: str | None = None, timeout_seconds: float = 1800.0) -> None:
        self.judge_url = judge_url
        self.timeout_seconds = float(timeout_seconds)
        self.token = _token()

    async def evaluate(self, problem_id: str, code: str) -> JudgeFeedback:
        url = (self.judge_url or os.environ.get(
            "FRONTIERCS_ALE_JUDGE_URL", "http://127.0.0.1:8082"
        )).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{url}/evaluate",
                    json={"problem_id": problem_id, "code": code},
                    headers={"Authorization": f"Bearer {self.token}"},
                )
                response.raise_for_status()
                return JudgeFeedback.from_dict(response.json())
        except Exception as exc:
            return JudgeFeedback(status="error", score=0.0, error=f"ALE-Bench judge request failed: {exc!r}")


class AleBenchRunner:
    def __init__(self) -> None:
        _token()  # Refuse to start an unauthenticated code-execution service.
        if not os.environ.get("ALE_BENCH_DATA"):
            raise RuntimeError("ALE_BENCH_DATA must point to official ALE-Bench ZIPs")
        self.num_workers = max(1, int(os.environ.get("ALE_BENCH_NUM_WORKERS", "4")))
        self.max_concurrent_evaluations = max(
            1, int(os.environ.get("ALE_BENCH_MAX_CONCURRENT_EVALUATIONS", "4"))
        )
        self.diagnostics_limit = max(0, int(os.environ.get("ALE_BENCH_DIAGNOSTICS_CHARS", "12000")))
        self.score_reference = max(1.0, float(os.environ.get("ALE_BENCH_SCORE_REFERENCE", "1000000000")))
        self._sessions: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self._evaluation_slots = threading.BoundedSemaphore(self.max_concurrent_evaluations)

    def _session(self, problem_id: str):
        with self._registry_lock:
            if problem_id not in self._sessions:
                from ale_bench import start

                self._sessions[problem_id] = start(
                    problem_id,
                    lite_version=False,
                    use_same_time_scale=False,
                    maximum_num_call_public_eval=10**18,
                    session_duration=timedelta(days=365),
                    num_workers=self.num_workers,
                    run_visualization_server=False,
                )
                self._locks[problem_id] = threading.Lock()
            return self._sessions[problem_id]

    def evaluate(self, problem_id: str, code: str) -> dict[str, Any]:
        try:
            with self._evaluation_slots:
                session = self._session(problem_id)
                with self._locks[problem_id]:
                    result = session.public_eval(
                        code, code_language="cpp20", skip_local_visualization=True,
                        reuse_containers=True,
                    )
        except Exception as exc:
            return JudgeFeedback(status="error", score=0.0,
                                 error=f"ALE-Bench evaluator error: {exc!r}").writer_payload()

        cases = list(result.case_results)
        counts = Counter(case.judge_result.value for case in cases)
        accepted = [case for case in cases if case.judge_result.value == "ACCEPTED"]
        metadata = session.problem.metadata
        mean_absolute = sum(float(case.absolute_score) for case in accepted) / len(cases) if cases else 0.0
        log_score = math.log1p(max(0.0, mean_absolute))
        log_reference = math.log1p(self.score_reference)
        if metadata.score_type.value == "maximize":
            # A logarithmic transform avoids saturating tasks whose raw scores
            # are naturally much smaller/larger than the shared reference.
            score = 100.0 * log_score / (log_score + log_reference)
        elif len(accepted) != len(cases):
            score = 0.0  # A failed case must not look like a good low-cost result.
        else:
            score = 100.0 * log_reference / (log_score + log_reference)

        lines = []
        for index, case in enumerate(cases):
            message, error = str(case.message or "").strip(), str(case.error_str or "").strip()
            if case.judge_result.value != "ACCEPTED" or message or error:
                line = f"case {index}: {case.judge_result.value}"
                if message:
                    line += f" — {message}"
                if error:
                    line += f" — {error}"
                lines.append(line)
        diagnostics = "\n".join(lines)
        truncated = len(diagnostics) > self.diagnostics_limit
        if truncated:
            diagnostics = diagnostics[: self.diagnostics_limit]

        overall = result.overall_judge_result.value
        status = "done"
        for ale_status, mapped in (
            ("COMPILATION_ERROR", "compile_error"),
            ("TIME_LIMIT_EXCEEDED", "time_limit_exceeded"),
            ("RUNTIME_ERROR", "runtime_error"),
            ("MEMORY_LIMIT_EXCEEDED", "memory_limit_exceeded"),
            ("WRONG_ANSWER", "wrong_answer"),
        ):
            if ale_status in counts:
                status = mapped
                break
        case_feedback = [{"summary": (
            f"{len(accepted)}/{len(cases)} public cases accepted; mean raw score per case="
            f"{mean_absolute:.3f}; statuses={dict(counts)}"
        )}]
        case_feedback.extend({"case_index": i, "judge_result": c.judge_result.value,
                              "absolute_score": int(c.absolute_score)}
                             for i, c in [(j, x) for j, x in enumerate(cases)
                                          if x.judge_result.value != "ACCEPTED"][:8])
        feedback = JudgeFeedback(
            status=status, score=score, score_unbounded=float(result.overall_absolute_score),
            passed=overall == "ACCEPTED", error=None if overall == "ACCEPTED" else overall,
            case_feedback=case_feedback, diagnostics=diagnostics, diagnostics_truncated=truncated,
            raw={"dataset": "ALE-Bench", "problem_id": problem_id,
                 "problem_type": metadata.problem_type.value, "score_type": metadata.score_type.value,
                 "overall_judge_result": overall, "accepted_cases": len(accepted),
                 "total_public_cases": len(cases), "case_status_counts": dict(counts),
                 "public_mean_absolute_score": mean_absolute,
                 "normalization_reference": self.score_reference},
        )
        return {**feedback.writer_payload(), "raw": feedback.raw}


class Handler(BaseHTTPRequestHandler):
    runner: AleBenchRunner

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, {"status": "ok", "judge": "ALE-Bench public_eval"}) if self.path == "/health" else self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/evaluate":
            self._send(404, {"error": "not found"})
            return
        expected = f"Bearer {_token()}"
        actual = self.headers.get("Authorization", "")
        if not hmac.compare_digest(actual, expected):
            self._send(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("request size must be between 1 and 2 MB")
            request = json.loads(self.rfile.read(length))
            problem_id, code = str(request["problem_id"]), str(request["code"])
            if not problem_id.replace("-", "").replace("_", "").isalnum():
                raise ValueError("invalid problem_id")
            self._send(200, self.runner.evaluate(problem_id, code))
        except Exception as exc:
            self._send(400, {"status": "error", "score": 0.0, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[ale-bench-judge] {format % args}", flush=True)


def serve() -> None:
    _token()
    host = os.environ.get("ALE_BENCH_JUDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("ALE_BENCH_JUDGE_PORT", "8082"))
    Handler.runner = AleBenchRunner()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ALE-Bench public judge listening on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    if not parser.parse_args().serve:
        parser.error("pass --serve")
    serve()
