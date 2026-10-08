#!/bin/bash
#SBATCH --job-name=mixeval
#SBATCH --nodes=1 --cpus-per-task=32 --mem=300G --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#
# 混合模型评测 (ACT 与 WRITE 解耦): ACT 与 WRITE 各自可选 本地 ckpt / 独立 sglang server / OpenAI 兼容 API / Anthropic API。
#   用法见同目录 README.md。在 examples/codebase_adaption 目录下提交 (日志落 logs/%x-%j.out); 分区/QOS/约束/GPU 数在命令行给。
#   集群相关路径全部走环境变量 (MILES_ROOT, MIX_SIF, MIX_ENV_FILE, MIX_EVAL_SCRIPT, MIX_INNER_SCRIPT), 默认值是 babel 的。
#   ACT:   ACT_API_MODEL 非空 -> ACT 走 API (EVAL_CKPT_DIR 仍需给一个本地 ckpt 占位: 评测链路要 Megatron+sglang 起来,
#          GPU 0-1 上的本地模型不会被调用); 为空 -> ACT 用 EVAL_CKPT_DIR 的本地模型。
#   WRITE: WRITE_MODE=api (WRITE_API_MODEL, 默认 gpt-5.5) | server (WRITE_HF 在 GPU 2 起 sglang) | local (与 ACT 同一本地模型)。
#   派生副本 codebase_rollout_mix_<jobid>.py (python 现场派生, 校验替换次数), 用 si3 的 CODEBASE_ROLLOUT_MODULE 选它, 跑完删。
#   API key 从 MIX_ENV_FILE (默认 $MILES_ROOT/.env) 读 OPENAI_API_KEY / ANTHROPIC_API_KEY (按 ACT_API_PROVIDER), 不写进文件/日志。
#   ACT_API_PROVIDER=anthropic 路径, 对齐官方 clbench ICL 的 Anthropic 原生 Messages 调用
#   (无 system, 全历史, JSON 指令追加在最后一条 user 文本后, max_tokens=128000, 倒数第二条消息 cache_control, 解析失败重采样 4 次).
#   ACT API 持续失败或上下文超限 (HTTP 400) -> 抛异常, 本 episode ABORTED (与本地超长同口径); WRITE API 失败 -> 本题记忆不更新。
# 需要 export: EVAL_MODE(stateless|replace) EVAL_CKPT_DIR EVAL_TAG; 可选 ACT_API_MODEL WRITE_MODE WRITE_API_MODEL WRITE_HF
#            API_MAX_TOKENS_ACT(32768) API_MAX_TOKENS_WRITE(16384) API_CONCURRENCY(8)
# 提交时按 WRITE_MODE 给 --gres=gpu:2 (api/local) 或 gpu:3 (server)。
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE PIP_CERT
set -uo pipefail

MILES_ROOT="${MILES_ROOT:-$HOME/qixinx/miles}"; M="$MILES_ROOT/examples/codebase_adaption"; export M MILES_ROOT
SIF="${MIX_SIF:-/data/group_data/rl/yuxiaoq/shared/assets/images/miles_dev-202606081341.sif}"   # 仅 WRITE_MODE=server 需要
ENV_FILE="${MIX_ENV_FILE:-$MILES_ROOT/.env}"; export ENV_FILE
EVAL_SCRIPT="${MIX_EVAL_SCRIPT:-$M/scripts/eval_ckpt_19q_babel2.sh}"          # 本集群的 19 题评测入口
INNER_SCRIPT="${MIX_INNER_SCRIPT:-$M/run_codebase_adaption_qwen3.5_4B_multiport_si3.sh}"   # 支持 CODEBASE_ROLLOUT_MODULE 的 run 脚本
export EVAL_MODE="${EVAL_MODE:?}" EVAL_CKPT_DIR="${EVAL_CKPT_DIR:?}" EVAL_TAG="${EVAL_TAG:?}"
export ACT_NOBUDGET="${ACT_NOBUDGET:-0}"   # 1: 去掉每轮 [Budget] 提示 (ablation A2)
export ACT_STATEFUL="${ACT_STATEFUL:-0}"   # 1: ACT 走 Responses API 有状态 (ablation A3)
export ACT_API_PROVIDER="${ACT_API_PROVIDER:-openai}"   # openai | anthropic
export ACT_JSON="${ACT_JSON:-0}"   # 1: ACT 走官方 JSON system 模板 + strict json_schema 结构化输出 (ablation A1)
export ROLLOUT_SRC="${ROLLOUT_SRC:-codebase_rollout.py}"   # 可选 codebase_rollout_icl.py (naive ICL: 19 题一条上下文, 配 CODEBASE_NO_MEMORY=1)
export ACT_API_MODEL="${ACT_API_MODEL:-}" WRITE_MODE="${WRITE_MODE:-local}" WRITE_API_MODEL="${WRITE_API_MODEL:-gpt-5.5}" WRITE_HF="${WRITE_HF:-}"
export API_MAX_TOKENS_ACT="${API_MAX_TOKENS_ACT:-32768}" API_MAX_TOKENS_WRITE="${API_MAX_TOKENS_WRITE:-16384}" API_CONCURRENCY="${API_CONCURRENCY:-8}"
MP_OFF=$(( SLURM_JOB_ID % 500 )); PORT=$(( 50000 + MP_OFF * 2 )); export WRITE_URL="http://127.0.0.1:${PORT}/generate"
GEN="$M/codebase_rollout_mix_${SLURM_JOB_ID}.py"; export GEN
SCR=/scratch/$USER; mkdir -p "$SCR/apptainer_tmp" "$SCR/tmp" "$M/logs"
export APPTAINER_TMPDIR=$SCR/apptainer_tmp TMPDIR=$SCR/tmp
export CODEBASE_APPTAINER_RUNTIME_ROOT="/scratch/${USER}/cb-rt-${SLURM_JOB_ID}"
WS=""; cleanup(){ [ -n "$WS" ] && kill $WS 2>/dev/null; rm -f "$GEN"; }; trap cleanup EXIT
LOG="${MIX_LOG:-$SLURM_SUBMIT_DIR/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out}"   # 须与 --output 一致 (watch 行要 grep 它)

echo "=== JOB $SLURM_JOB_ID on $(hostname)  $(date) ==="
echo "    MODE=$EVAL_MODE  TAG=$EVAL_TAG  ACT=${ACT_API_MODEL:-local:$EVAL_CKPT_DIR}  WRITE=$WRITE_MODE ${WRITE_API_MODEL} ${WRITE_HF}  GPUs=${CUDA_VISIBLE_DEVICES:-}"
_STAG=$(( (SLURM_JOB_ID % 7) * 45 )); echo "=== stagger ${_STAG}s ==="; sleep ${_STAG}

[ -f "$EVAL_CKPT_DIR/latest_checkpointed_iteration.txt" ] || { echo "!!! EVAL_CKPT_DIR 无效: $EVAL_CKPT_DIR"; exit 1; }
if [ -n "$ACT_API_MODEL" ] || [ "$WRITE_MODE" = api ]; then
  if [ "$ACT_API_PROVIDER" = anthropic ]; then KEYNAME=ANTHROPIC_API_KEY; HOSTN=api.anthropic.com; else KEYNAME=OPENAI_API_KEY; HOSTN=api.openai.com; fi
  grep -q "^${KEYNAME}=.\+" "$ENV_FILE" || { echo "!!! $ENV_FILE 里没有 $KEYNAME"; exit 1; }
  curl -s --max-time 20 -o /dev/null -w "    $HOSTN 连通性 HTTP %{http_code} (401 = 可达)\n" https://$HOSTN/v1/models || echo "!!! 计算节点连不上 $HOSTN"
fi
[ "$WRITE_MODE" = server ] && { [ -f "$WRITE_HF/config.json" ] || { echo "!!! WRITE_HF 无效: $WRITE_HF"; exit 1; }; }
echo "=== 路径校验通过 ==="

# 1. 派生 rollout 副本
python3 - <<'PYEOF'
import os, re, ast
M=os.environ["M"]
SRC=os.environ.get("ROLLOUT_SRC","codebase_rollout.py"); src=open(f"{M}/{SRC}").read(); out=src; notes=[f"src={SRC}"]
if os.environ.get("ACT_API_MODEL"):
    old=("                act_text, _resp_ids, _resp_lps, act_finish = await _infer(\n"
         "                    url,\n"
         "                    prompt_ids,\n"
         "                    _structured_act_params(sampling_params, query),\n"
         "                )\n")
    assert out.count(old)==1, "ACT _infer 块不唯一/不存在, 原文件结构已变"
    out=out.replace(old,"                act_text, _resp_ids, _resp_lps, act_finish = await _apiact_infer(state, messages)\n")
    notes.append("ACT->api")
wm=os.environ.get("WRITE_MODE","local")
wline="await _infer(url, write_ids, write_params)"
assert out.count(wline)==1, "WRITE _infer 行不唯一"
if wm=="api":
    out=out.replace(wline,"await _apiwrite_infer(state, write_messages, write_params)"); notes.append("WRITE->api")
elif wm=="server":
    uline='    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"\n'
    assert out.count(uline)==1
    out=out.replace(uline, uline+f'    write_url = "{os.environ["WRITE_URL"]}"\n')
    out=out.replace(wline,"await _infer(write_url, write_ids, write_params)"); notes.append("WRITE->server")
if os.environ.get("ACT_JSON")=="1":
    l1='    _cb_task_mod._SYSTEM_TEMPLATE = _TEXT_ACT_SYSTEM_TEMPLATE\n'
    assert out.count(l1)==1, "文本格式覆盖行不唯一"
    out=out.replace(l1,'    pass  # ACT_JSON=1: 保留 clbench 官方 JSON system 模板 (thought/command)\n')
    l2='action = _response_from_text(query, act_visible)'
    assert out.count(l2)==1, "_response_from_text 调用不唯一"
    out=out.replace(l2,'action = _response_from_text_json(query, act_visible)')
    notes.append("ACT->json-structured-output")
if os.environ.get("ACT_NOBUDGET")=="1":
    b0='                user_content = (\n                    f"{user_content}\\n\\n"\n'
    assert out.count(b0)==1, "[Budget] 块开头不唯一"
    st=out.index(b0); en=out.index('                )\n',st)+len('                )\n')
    assert '[Budget] You have' in out[st:en], "[Budget] 块定位错误"
    out=out[:st]+'                pass  # ACT_NOBUDGET=1: 去掉每轮 [Budget] 剩余轮数提示 (ablation A2)\n'+out[en:]
    notes.append("ACT->no-budget-line")
if os.environ.get("ACT_STATEFUL")=="1":
    notes.append("ACT->responses-api-stateful")
helpers=r'''


# ---- appended by scripts/mixmodel/mixmodel_eval.sbatch: ACT/WRITE via OpenAI-compatible chat API ----
import time as _api_time
import requests as _api_requests
_API_BASE = "https://api.openai.com/v1"
_API_ENV = "__ENV_FILE__"
_API_PROVIDER = "__PROVIDER__"   # openai | anthropic (ACT 的提供方; WRITE 仍走 OpenAI)
_API_ACT_MODEL = "__ACT_MODEL__"
_API_WRITE_MODEL = "__WRITE_MODEL__"
_API_MAX_ACT = __MAX_ACT__
_API_MAX_WRITE = __MAX_WRITE__
_API_CONC = __CONC__
_API_JSON = __JSON__   # True: ACT 用 strict json_schema 结构化输出 (对齐官方 clbench ICL), 历史里存 JSON 原文
_API_JSON_SCHEMA = {"type": "object",
                    "properties": {"thought": {"type": "string", "description": "Your reasoning and analysis. Explain why you want to perform the action."},
                                   "command": {"type": "string", "description": "The bash command to execute. ONLY the raw command, no markdown fences or explanation."}},
                    "required": ["thought", "command"], "additionalProperties": False}
_API_STATEFUL = __STATEFUL__   # True: ACT 走 OpenAI Responses API, store=true + previous_response_id (reasoning 跨轮保留, 对齐官方 clbench)
_API_STATE = {}                # id(messages) -> {"prev": response_id, "sent": 已发送的 messages 长度}
_API_KEYS = {}
_API_SEM = None


def _api_key(name="OPENAI_API_KEY"):
    if name not in _API_KEYS:
        for line in open(_API_ENV):
            line = line.strip()
            if line.startswith(name + "="):
                _API_KEYS[name] = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not _API_KEYS.get(name):
            raise RuntimeError(f"{name} not found in .env")
    return _API_KEYS[name]


class _ApiFatal(Exception):
    pass


def _api_chat(tag, model, messages, max_tokens, attempts, json_schema=None):
    payload = {"model": model, "messages": messages, "max_completion_tokens": int(max_tokens)}
    if json_schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "BashCommandResponse", "strict": True, "schema": json_schema}}
    for attempt in range(attempts):
        try:
            r = _api_requests.post(_API_BASE + "/chat/completions",
                                   headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
                                   json=payload, timeout=1200)
            if r.status_code == 400:
                body = (r.text or "")[:300]
                print(f"[{tag}] FATAL HTTP 400 (不可重试, 多为上下文超限): {body}", flush=True)
                raise _ApiFatal(body)
            r.raise_for_status()
            j = r.json(); ch = j["choices"][0]
            text = ch["message"].get("content") or ""
            u = j.get("usage", {})
            print(f"[{tag}] OK model={j.get('model')} attempt={attempt} prompt_tokens={u.get('prompt_tokens')} "
                  f"completion_tokens={u.get('completion_tokens')} resp_chars={len(text)} finish={ch.get('finish_reason')}", flush=True)
            return text, (ch.get("finish_reason") or "stop")
        except _ApiFatal:
            raise
        except Exception as e:
            body = ""
            try:
                body = (getattr(getattr(e, "response", None), "text", "") or "")[:200]
            except Exception:
                pass
            print(f"[{tag}] ERROR attempt={attempt} {type(e).__name__}: {e} {body}", flush=True)
            _api_time.sleep(min(15 * (attempt + 1), 120))
    return None, "abort"


def _api_responses(tag, model, items, prev_id, json_schema, attempts):
    """Responses API 一轮: 只发新消息 + previous_response_id. 返回 (text, finish, response_id, chain_ok)."""
    payload = {"model": model, "input": items, "store": True}
    if prev_id:
        payload["previous_response_id"] = prev_id
    if json_schema is not None:
        payload["text"] = {"format": {"type": "json_schema", "name": "BashCommandResponse", "schema": json_schema, "strict": True}}
    for attempt in range(attempts):
        try:
            r = _api_requests.post(_API_BASE + "/responses",
                                   headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
                                   json=payload, timeout=1200)
            if r.status_code == 400:
                body = (r.text or "")[:300]
                if prev_id and ("previous_response" in body or "not found" in body.lower()):
                    print(f"[{tag}] chain broken (400 on previous_response_id), will resend full history: {body}", flush=True)
                    return None, "abort", None, False
                print(f"[{tag}] FATAL HTTP 400 (不可重试, 多为上下文超限): {body}", flush=True)
                raise _ApiFatal(body)
            r.raise_for_status()
            j = r.json()
            texts = []
            for it in j.get("output") or []:
                if it.get("type") == "message":
                    for c in it.get("content") or []:
                        if c.get("type") == "output_text":
                            texts.append(c.get("text") or "")
            text = "".join(texts)
            status = j.get("status")
            reason = ((j.get("incomplete_details") or {}).get("reason") or "")
            finish = "length" if (status == "incomplete" and "max_output" in reason) else "stop"
            u = j.get("usage", {})
            print(f"[{tag}] OK model={j.get('model')} attempt={attempt} prompt_tokens={u.get('input_tokens')} "
                  f"completion_tokens={u.get('output_tokens')} resp_chars={len(text)} finish={status}/{reason} chained={bool(prev_id)}", flush=True)
            return text, finish, j.get("id"), True
        except _ApiFatal:
            raise
        except Exception as e:
            body = ""
            try:
                body = (getattr(getattr(e, "response", None), "text", "") or "")[:200]
            except Exception:
                pass
            print(f"[{tag}] ERROR attempt={attempt} {type(e).__name__}: {e} {body}", flush=True)
            _api_time.sleep(min(15 * (attempt + 1), 120))
    return None, "abort", None, True

_ANTH_INSTR = None


def _anth_instruction():
    """官方 clbench ICL 对 Anthropic 的做法: 把 schema_to_prompt_instruction(BashCommandResponse) 追加到最后一条 user 文本后 (不进历史)."""
    global _ANTH_INSTR
    if _ANTH_INSTR is None:
        try:
            import src.tasks.codebase_adaptation.task as _t
            schema = _t.BashCommandResponse.model_json_schema()
        except Exception as e:
            print(f"[apiact] WARN 无法导入官方 BashCommandResponse ({e}), 用内置 schema", flush=True)
            schema = {"description": "Structured response for codebase adaptation task.",
                      "properties": {"thought": {"description": "Your reasoning and analysis. Explain why you want to perform this action.", "title": "Thought", "type": "string"},
                                     "command": {"description": "The bash command to execute. ONLY the raw command, no markdown fences or explanation.", "title": "Command", "type": "string"}},
                      "required": ["thought", "command"], "title": "BashCommandResponse", "type": "object"}
        _ANTH_INSTR = ("\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
                       f"```json\n{json.dumps(schema, indent=2)}\n```\n"
                       "Respond ONLY with the JSON object, no other text.")
    return _ANTH_INSTR


def _anth_parse_ok(text):
    """对齐官方 validate_with_coercion 的宽松解析: 裸 JSON / ```json 围栏 / 首个对象; 两个字段都要是字符串."""
    obj = _parse_json_object(text or "", {"thought", "command"})
    return isinstance(obj, dict) and "command" in obj and "thought" in obj


def _api_anthropic(tag, model, messages, max_tokens, attempts):
    """Anthropic Messages API (对齐官方 provider_adapters._anthropic_structured): 无 system, 全历史, 指令追加在最后 user,
    倒数第二条消息打 cache_control, 无 thinking/temperature. 返回 (text, finish). 解析失败按官方重采样 (最多 4 次)."""
    msgs = [{"role": m["role"], "content": [{"type": "text", "text": str(m.get("content", ""))}]}
            for m in messages if m.get("role") != "system"]
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": [{"type": "text", "text": ""}]})
    msgs[-1]["content"][-1]["text"] += _anth_instruction()
    if len(msgs) >= 2:
        msgs[-2]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    payload = {"model": model, "max_tokens": int(max_tokens), "messages": msgs}
    hdr = {"x-api-key": _api_key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01", "content-type": "application/json"}
    last = (None, "abort")
    for parse_try in range(4):
        text = None
        for attempt in range(attempts):
            try:
                r = _api_requests.post("https://api.anthropic.com/v1/messages", headers=hdr, json=payload, timeout=900)
                if r.status_code in (408, 425, 429) or r.status_code >= 500:
                    body = (r.text or "")[:200]
                    base = 15.0 if (r.status_code == 429 or "overloaded" in body.lower() or "rate" in body.lower()) else 2.0
                    wait = min(base * (2 ** attempt), 240)
                    print(f"[{tag}] ERROR attempt={attempt} HTTP {r.status_code} retry in {wait:.0f}s: {body}", flush=True)
                    _api_time.sleep(wait); continue
                if r.status_code >= 400:
                    body = (r.text or "")[:300]
                    print(f"[{tag}] FATAL HTTP {r.status_code} (不可重试, 多为上下文超限): {body}", flush=True)
                    raise _ApiFatal(body)
                j = r.json()
                parts = [b.get("text", "") for b in j.get("content", []) if isinstance(b, dict) and b.get("type") == "text"]
                if not parts:
                    raise RuntimeError("Anthropic response did not include text content")
                text = "\n".join(parts)
                u = j.get("usage", {})
                finish = "length" if j.get("stop_reason") == "max_tokens" else "stop"
                print(f"[{tag}] OK model={j.get('model')} attempt={attempt} prompt_tokens={u.get('input_tokens')} "
                      f"cache_read={u.get('cache_read_input_tokens')} cache_write={u.get('cache_creation_input_tokens')} "
                      f"completion_tokens={u.get('output_tokens')} resp_chars={len(text)} finish={j.get('stop_reason')} parse_try={parse_try}", flush=True)
                break
            except _ApiFatal:
                raise
            except Exception as e:
                print(f"[{tag}] ERROR attempt={attempt} {type(e).__name__}: {str(e)[:200]}", flush=True)
                _api_time.sleep(min(15 * (attempt + 1), 120))
        if text is None:
            return last
        last = (text, finish)
        if _anth_parse_ok(text):
            return last
        print(f"[{tag}] PARSEFAIL parse_try={parse_try} resp_chars={len(text)} head={text[:120]!r}; 按官方口径重采样", flush=True)
        _api_time.sleep(0.5 * (2 ** parse_try))
    return last


def _api_sem():
    global _API_SEM
    if _API_SEM is None:
        _API_SEM = asyncio.Semaphore(_API_CONC)
    return _API_SEM


async def _apiact_infer(state, messages):
    async with _api_sem():
        if _API_PROVIDER == "anthropic":
            text, finish = await asyncio.to_thread(_api_anthropic, "apiact", _API_ACT_MODEL, messages, _API_MAX_ACT, 8)
        elif _API_STATEFUL:
            key = id(messages); st = _API_STATE.setdefault(key, {"prev": None, "sent": 0})
            def _items(frm):
                return [{"role": m["role"], "content": str(m.get("content", ""))} for m in messages[frm:] if m.get("role") != "system"]
            schema = _API_JSON_SCHEMA if _API_JSON else None
            text, finish, rid, ok = await asyncio.to_thread(_api_responses, "apiact", _API_ACT_MODEL, _items(st["sent"]) if st["prev"] else _items(0), st["prev"], schema, 6)
            if not ok:   # 链断了: 整段历史重发, 不带 previous_response_id
                text, finish, rid, ok = await asyncio.to_thread(_api_responses, "apiact", _API_ACT_MODEL, _items(0), None, schema, 6)
            if rid:
                st["prev"] = rid; st["sent"] = len(messages) + 1   # +1: 调用方随后会把 assistant 消息追加进 messages, 不必重发
        else:
            text, finish = await asyncio.to_thread(_api_chat, "apiact", _API_ACT_MODEL, messages, _API_MAX_ACT, 6,
                                                   _API_JSON_SCHEMA if _API_JSON else None)
    if text is None:
        raise RuntimeError("apiact: API failed after retries -> abort episode")
    text = _strip_lone_surrogates(text)
    ids = state.tokenizer.encode(text, add_special_tokens=False) if text else []
    return text, ids, [0.0] * len(ids), ("length" if finish == "length" else "stop")


def _response_from_text_json(query, text):
    """ACT_JSON 模式: 回复是 {"thought","command"} JSON (与官方 clbench 一致); 解析失败回落到文本块解析。"""
    action_cls = query.response_schema
    fields = getattr(action_cls, "model_fields", {})
    try:
        obj = json.loads(text or "")
        assert isinstance(obj, dict)
    except Exception:
        obj = _parse_json_object(text or "", set(fields))   # 围栏/前后缀文字/首个对象 (对齐官方 validate_with_coercion 的回落)
    if isinstance(obj, dict) and any(k in obj for k in fields):
        try:
            payload = {k: _strip_lone_surrogates(str(obj.get(k, "") or "")) for k in fields}
            return action_cls(**payload)
        except Exception:
            pass
    return _response_from_text(query, text)


async def _apiwrite_infer(state, messages, write_params):
    async with _api_sem():
        try:
            text, finish = await asyncio.to_thread(_api_chat, "apiwrite", _API_WRITE_MODEL, messages, _API_MAX_WRITE, 4)
        except _ApiFatal:
            text, finish = None, "abort"
    if text is None:
        print("[apiwrite] ERROR giving up -> memory unchanged for this issue", flush=True)
        text = ""
    text = _strip_lone_surrogates(text)
    ids = state.tokenizer.encode(text, add_special_tokens=False) if text else []
    return text, ids, [0.0] * len(ids), ("length" if finish == "length" else "stop")
'''
helpers=(helpers.replace("__ACT_MODEL__",os.environ.get("ACT_API_MODEL") or "")
         .replace("__PROVIDER__",os.environ.get("ACT_API_PROVIDER","openai"))
         .replace("__ENV_FILE__",os.environ["ENV_FILE"])
         .replace("__WRITE_MODEL__",os.environ.get("WRITE_API_MODEL","gpt-5.5"))
         .replace("__MAX_ACT__",os.environ.get("API_MAX_TOKENS_ACT","32768"))
         .replace("__MAX_WRITE__",os.environ.get("API_MAX_TOKENS_WRITE","16384"))
         .replace("__CONC__",os.environ.get("API_CONCURRENCY","8"))
         .replace("__JSON__","True" if os.environ.get("ACT_JSON")=="1" else "False")
         .replace("__STATEFUL__","True" if os.environ.get("ACT_STATEFUL")=="1" else "False"))
out+=helpers
ast.parse(out)
assert "sk-" not in out, "副本里出现疑似密钥"
open(os.environ["GEN"],"w").write(out)
print("=== 派生校验通过:", ", ".join(notes) or "无替换 (ACT/WRITE 都本地)", "->", os.environ["GEN"])
PYEOF
[ $? -eq 0 ] || { echo "!!! 派生失败"; exit 1; }
diff "$M/${ROLLOUT_SRC:-codebase_rollout.py}" "$GEN" | grep -E '^[<>]' | grep -vE '^> *$' | head -12

# 2. WRITE server (可选)
if [ "$WRITE_MODE" = server ]; then
  echo "=== 启动 WRITE server (GPU 2, port $PORT) ==="; WLOG="$M/logs/${EVAL_TAG}-writeserver.log"
  apptainer exec --nv --bind /data,/scratch --env CUDA_VISIBLE_DEVICES=2 "$SIF" \
    python3 -m sglang.launch_server --model-path "$WRITE_HF" --host 127.0.0.1 --port "$PORT" \
      --tp-size 1 --mem-fraction-static 0.85 --trust-remote-code > "$WLOG" 2>&1 &
  WS=$!
  for i in $(seq 1 90); do sleep 20
    curl -s --max-time 5 "http://127.0.0.1:$PORT/get_model_info" | grep -q model_path && { echo "=== WRITE server 就绪 ($((i*20))s) ==="; READY=1; break; }
    kill -0 $WS 2>/dev/null || { echo "!!! WRITE server 已死"; tail -30 "$WLOG"; exit 1; }
  done
  [ "${READY:-}" = 1 ] || { echo "!!! WRITE server 30min 未就绪"; exit 1; }
fi

# 3. eval: 本地 ACT/占位模型占 GPU 0-1, 走本集群评测入口 + 支持 CODEBASE_ROLLOUT_MODULE 的 run 脚本, 用派生 rollout 模块
export CUDA_VISIBLE_DEVICES=0,1
export CODEBASE_NGPU=2 CODEBASE_TP=2 CODEBASE_SGLANG_MEM=0.5
export CODEBASE_ADDR_ALLOC_PATCH=1
export CODEBASE_INNER_SCRIPT=$INNER_SCRIPT
export CODEBASE_ROLLOUT_MODULE=examples.codebase_adaption.codebase_rollout_mix_${SLURM_JOB_ID}
export CODEBASE_ACT_THINKING=1 CODEBASE_WRITE_THINKING=1
export CODEBASE_WRITE_PROMPT_V3=1 CODEBASE_MULTIBLOCK_FEEDBACK=1   # 训练开关, 逐项照搬
[ "$EVAL_MODE" = stateless ] && export CODEBASE_EVAL_PROMPT_DATA="$M/data/baseline19_textfmt_eval_think.jsonl"
if [ "$EVAL_MODE" = stateless ]; then EXPECT=76; else EXPECT=5; fi

TRAJ="$HOME/eval_traj/eval-${EVAL_TAG}-${EVAL_MODE}/traj"
if [ -d "$TRAJ/eval" ] && [ "$(find "$TRAJ/eval" -name 'ep_*.json' 2>/dev/null | wc -l)" -gt 0 ]; then
  mv "$TRAJ" "$TRAJ.prev-$(date +%m%d%H%M)"; echo "=== 已归档陈旧轨迹 ==="
fi
mkdir -p "$TRAJ"
( while true; do sleep 60
    n=$(find "$TRAJ/eval" -name 'ep_*.json' 2>/dev/null | wc -l)
    echo "[watch $(date +%H:%M:%S)] eval episodes: $n / $EXPECT   apiact OK/ERR/FATAL: $(grep -c '\[apiact\] OK' "$LOG")/$(grep -c '\[apiact\] ERROR' "$LOG")/$(grep -c '\[apiact\] FATAL' "$LOG")   apiwrite OK/ERR: $(grep -c '\[apiwrite\] OK' "$LOG")/$(grep -c '\[apiwrite\] ERROR' "$LOG")   write-server POST: $( [ -n "${WLOG:-}" ] && grep -c 'POST /generate' "$WLOG" || echo -)"
    [ "$n" -ge "$EXPECT" ] && { echo "[watch] eval 全部落盘, 60s 后收尾"; sleep 60; cleanup; scancel "$SLURM_JOB_ID"; break; }
  done ) & WATCH=$!

bash "$EVAL_SCRIPT"; RC=$?
kill $WATCH 2>/dev/null
echo "=== rc=$RC  episodes=$(find "$TRAJ/eval" -name 'ep_*.json' 2>/dev/null | wc -l)/$EXPECT  $(date) ==="
