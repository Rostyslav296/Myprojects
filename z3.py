#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anacharssis_Z3 — Minimal AI Agent with Tools + LLM (Ollama)
Owner: Rostyslav
Version: Aug 2025 (Ollama chat, NL routing, working web.search)
"""

# ===================== Imports & Setup =====================
import os, re, shlex, json, math, ast, secrets, string, pathlib, subprocess, shutil, sys, time
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple
from urllib import request, parse, error as urlerror
from html import unescape

HOME = pathlib.Path.home().resolve()
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5")
_HINT_SHOWN = False

def _norm(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = (pathlib.Path.cwd() / p).resolve()
    return p

# ===================== Tools Framework =====================
@dataclass
class ToolSpec:
    name: str
    fn: Callable
    help: str = ""
    args: Dict[str, Tuple[str, object]] = field(default_factory=dict)  # {name: (type, default|'...')}
    guard: Optional[Callable] = None
    tags: Tuple[str, ...] = ("general",)

TOOLS: Dict[str, ToolSpec] = {}

def register_tool(name: str, *, help: str, args: Dict[str, Tuple[str, object]] | None = None,
                  guard: Optional[Callable] = None, tags: Tuple[str, ...] = ("general",)):
    def deco(fn: Callable):
        TOOLS[name] = ToolSpec(name=name, fn=fn, help=help, args=args or {}, guard=guard, tags=tags)
        return fn
    return deco

_BOOL_TRUE = {"1","true","yes","on","y","ok"}

def _cast_arg(v: str, ty: str):
    if ty == "str": return v
    if ty == "int": return int(v)
    if ty == "float": return float(v)
    if ty == "bool": return v.strip().lower() in _BOOL_TRUE
    if ty == "path": return str(_norm(v))
    if ty == "json":
        try: return json.loads(v)
        except Exception as e: raise ValueError(f"invalid json: {e}")
    raise ValueError(f"unknown type '{ty}'")

def _normalize_args(spec: ToolSpec, pos_args: List[str], kv_args: Dict[str,str]):
    out = {}
    ordered = list(spec.args.items())  # preserve declared order
    for i, (k,(ty,default)) in enumerate(ordered):
        if i < len(pos_args):
            out[k] = _cast_arg(pos_args[i], ty)
        elif k in kv_args:
            out[k] = _cast_arg(kv_args[k], ty)
        else:
            if default == "...":
                raise ValueError(f"missing required arg '{k}'")
            out[k] = default
    extras = set(kv_args) - set(spec.args)
    if extras:
        raise ValueError("unknown args: " + ", ".join(sorted(extras)))
    return out

def parse_tool_invocation(s: str) -> Tuple[str, List[str], Dict[str,str]]:
    # expects: "tool <name> [pos...] [k=v ...]"
    parts = shlex.split(s)
    if len(parts) < 2: raise ValueError("usage: tool <name> [args...]")
    name = parts[1]
    pos, kv = [], {}
    for tok in parts[2:]:
        if "=" in tok:
            k, v = tok.split("=",1); kv[k] = v
        else:
            pos.append(tok)
    return name, pos, kv

def run_tool_std(name: str, *, pos_args: List[str] | None = None, kv_args: Dict[str,str] | None = None) -> str:
    pos_args = pos_args or []; kv_args = kv_args or {}
    spec = TOOLS.get(name)
    if not spec: return f"[tool_error: unknown tool '{name}']"
    try:
        args = _normalize_args(spec, pos_args, kv_args)
    except Exception as e:
        return f"[arg_error:{e}]"
    if spec.guard:
        err = spec.guard(args)
        if err: return err
    try:
        return spec.fn(**args)
    except Exception as e:
        return f"[tool_runtime_error:{e}]"

def tools_list() -> str:
    if not TOOLS: return "(no tools registered)"
    groups: Dict[str,List[str]] = {}
    for t in TOOLS.values():
        for tag in t.tags:
            groups.setdefault(tag, []).append(t.name)
    lines = []
    for tag in sorted(groups):
        lines.append(f"[{tag}]")
        for n in sorted(groups[tag]): lines.append(f"  - {n}")
    lines.append("\nTry: tools help <name>  or  tool <name> key=val")
    return "\n".join(lines)

def tools_help(name: Optional[str] = None) -> str:
    if not name: return tools_list()
    spec = TOOLS.get(name)
    if not spec: return f"(no such tool: {name})"
    def fmt_default(d): return "required" if d == "..." else json.dumps(d)
    arglines = [f"  {k}: {ty}, default={fmt_default(d)}" for k,(ty,d) in spec.args.items()]
    tags = ", ".join(spec.tags) if spec.tags else "general"
    return f"{name} — {spec.help}\nTags: {tags}\nArgs:\n" + ("\n".join(arglines) if arglines else "  (none)")

# ===================== Tools Implementation =====================
# ---------- File tools ----------
def _home_guard(args):
    p_raw = args.get("path","")
    if not p_raw: return None
    try:
        r = _norm(p_raw).resolve()
        if not (HOME in r.parents or r == HOME):
            return f"[guard: path must be inside HOME: {HOME}]"
    except Exception:
        return "[guard: invalid path]"
    return None

@register_tool("file.read", help="Read a UTF-8 text file.",
    args={"path": ("path","...")}, guard=_home_guard, tags=("files",))
def _tool_file_read(path: str) -> str:
    try: return _norm(path).read_text(encoding="utf-8")
    except Exception as e: return f"[read_error:{e}]"

@register_tool("file.write", help="Write text to a file (overwrite).",
    args={"path": ("path","..."), "text": ("str","...")}, guard=_home_guard, tags=("files",))
def _tool_file_write(path: str, text: str) -> str:
    try:
        p = _norm(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return f"OK: wrote {len(text)} bytes to {p}"
    except Exception as e:
        return f"[write_error:{e}]"

@register_tool("file.create", help="Create a new file (fails if exists). Seed with text if given.",
    args={"path": ("path","..."), "text": ("str","")}, guard=_home_guard, tags=("files",))
def _tool_file_create(path: str, text: str = "") -> str:
    try:
        p = _norm(path)
        if p.exists(): return f"[create_error: file exists: {p}]"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding="utf-8")
        return f"OK: created {p} ({len(text)} bytes)"
    except Exception as e:
        return f"[create_error:{e}]"

@register_tool("file.edit", help="Edit file. Modes: append, overwrite, insert(line=1-based), replace(pattern=regex).",
    args={"path": ("path","..."), "mode": ("str","append"), "text": ("str",""), "line": ("int",0), "pattern": ("str","")},
    guard=_home_guard, tags=("files",))
def _tool_file_edit(path: str, mode: str="append", text: str="", line: int=0, pattern: str="") -> str:
    try:
        p = _norm(path)
        if not p.exists(): return f"[edit_error: file not found: {p}]"
        content = p.read_text(encoding="utf-8")
        m = mode.lower()
        if m == "overwrite":
            p.write_text(text, encoding="utf-8"); return f"OK: overwrote {p} ({len(text)} bytes)"
        if m == "append":
            newc = content + ("" if content.endswith("\n") else "\n") + text
            p.write_text(newc, encoding="utf-8"); return f"OK: appended {len(text)} bytes to {p}"
        if m == "insert":
            lines = content.splitlines(True)
            idx = max(0, min((line-1) if line>0 else len(lines), len(lines)))
            lines.insert(idx, text if text.endswith("\n") else text + "\n")
            p.write_text("".join(lines), encoding="utf-8"); return f"OK: inserted at line {idx+1} in {p}"
        if m == "replace":
            if not pattern: return "[edit_error: replace mode requires pattern=...]"
            newc, n = re.subn(pattern, text, content, flags=re.MULTILINE)
            p.write_text(newc, encoding="utf-8"); return f"OK: replaced {n} occurrence(s) in {p}"
        return f"[edit_error: unknown mode '{mode}']"
    except Exception as e:
        return f"[edit_error:{e}]"

@register_tool("file.search", help="Search for files matching a glob (e.g. **/*.py).",
    args={"pattern": ("str","...")}, tags=("files","search"))
def _tool_file_search(pattern: str) -> str:
    try:
        matches = set()
        for base in (pathlib.Path.cwd(), HOME):
            for m in base.glob(pattern): matches.add(str(m))
        if not matches: return "No files found."
        return "Found files:\n" + "\n".join(f"  - {r}" for r in sorted(matches))
    except Exception as e:
        return f"[search_error:{e}]"

@register_tool("file.list", help="List files in a directory.",
    args={"path": ("path",".")}, tags=("files",))
def _tool_file_list(path: str) -> str:
    try:
        p = _norm(path)
        if not p.exists(): return f"[error: path does not exist: {p}]"
        if not p.is_dir(): return f"[error: not a directory: {p}]"
        files, dirs = [], []
        for item in p.iterdir():
            (dirs if item.is_dir() else files).append(item.name)
        lines = [f"Contents of {p}:"]
        for d in sorted(dirs):  lines.append(f"  [DIR]  {d}/")
        for f in sorted(files): lines.append(f"  [FILE] {f}")
        return "\n".join(lines)
    except Exception as e:
        return f"[scan_error:{e}]"

# ---------- System / Network ----------
@register_tool("usb.list", help="List USB devices (lsusb).",
    args={}, tags=("hardware","usb"))
def _tool_usb_list() -> str:
    try:
        if shutil.which("lsusb") is None:
            return "[usb_error: 'lsusb' not found. Install usbutils or run on a system that has it.]"
        result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return "Connected USB devices:\n" + result.stdout.strip()
        return "[error: No output from lsusb]"
    except Exception as e:
        return f"[usb_error:{e}]"

@register_tool("wifi.scan", help="Scan Wi-Fi (nmcli).",
    args={}, tags=("wifi","network"))
def _tool_wifi_scan() -> str:
    try:
        if shutil.which("nmcli") is None:
            return "[wifi_error: 'nmcli' not found. Install NetworkManager or run on a system that has it.]"
        result = subprocess.run(["nmcli","device","wifi","list"], capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            return "Available Wi-Fi networks:\n" + result.stdout.strip()
        return "[error: Wi-Fi scan failed. Is NetworkManager running?]"
    except Exception as e:
        return f"[wifi_error:{e}]"

# ---------- Utility / Demo ----------
@register_tool("demo.calc", help="Evaluate a basic math expression (digits + + - * / ( ) .).",
    args={"expr": ("str","...")}, tags=("math","demo"))
def _tool_demo_calc(expr: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr): return "[calc_error: invalid characters]"
        node = ast.parse(expr, mode="eval"); code = compile(node, "<calc>", "eval")
        return f"{expr} = {eval(code, {'__builtins__': {}}, vars(math))}"
    except Exception as e:
        return f"[calc_error:{e}]"

@register_tool("pass.gen", help="Generate a secure random password.",
    args={"length": ("int", 16), "symbols": ("bool", True)}, tags=("security","password"))
def _tool_pass_gen(length: int = 16, symbols: bool = True) -> str:
    try:
        length = max(4, min(int(length), 128))
        chars = string.ascii_letters + string.digits
        if symbols: chars += "!@#$%^&*"
        return ''.join(secrets.choice(chars) for _ in range(length))
    except Exception as e:
        return f"[pass_gen_error:{e}]"

# ---------- Web Search (DuckDuckGo HTML, no API key) ----------
def _http_get(url: str, data: Optional[bytes]=None, timeout: int=12) -> str:
    req = request.Request(url, data=data, headers={"User-Agent":"Mozilla/5.0 (Z3Agent)"})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")

@register_tool("web.search", help="Search the web (DuckDuckGo). Returns top results.",
    args={"query": ("str","..."), "max": ("int", 5)}, tags=("web","search"))
def _tool_web_search(query: str, max: int = 5) -> str:
    try:
        q = parse.urlencode({"q": query})
        html = _http_get("https://duckduckgo.com/html/?" + q)
        # crude parse of results
        items = []
        for m in re.finditer(r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>', html, re.I|re.S):
            href = unescape(m.group(1))
            title = re.sub("<.*?>","",unescape(m.group(2))).strip()
            if title and href:
                items.append((title, href))
            if len(items) >= max: break
        if not items:
            return f"No results for: {query}"
        out = [f"Top results for: {query}"]
        for i,(t,u) in enumerate(items,1):
            out.append(f"{i}. {t}\n   {u}")
        return "\n".join(out)
    except urlerror.URLError as e:
        return f"[web_error: network issue — {e}]"
    except Exception as e:
        return f"[web_error:{e}]"

# ---------- LLM (Ollama) ----------
def _ollama_request(path: str, payload: dict, timeout: int=60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    url = f"{OLLAMA_BASE}{path}"
    req = request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="ignore")
            # /api/chat streams by lines if stream=true. We send stream=false, so plain JSON.
            return json.loads(txt)
    except Exception as e:
        raise RuntimeError(str(e))

def _ollama_alive() -> bool:
    try:
        req = request.Request(f"{OLLAMA_BASE}/api/tags")
        req.add_header("Accept","application/json")
        with request.urlopen(req, timeout=4) as resp:
            _ = resp.read()
        return True
    except Exception:
        return False

def llm_chat(messages: List[dict], model: Optional[str]=None, temperature: float=0.2, max_tokens: int=512) -> str:
    if not _ollama_alive():
        return ("[llm_error: Ollama is not reachable at "
                f"{OLLAMA_BASE}. Start it:\n  $ ollama serve\n"
                f"and pull a model:\n  $ ollama pull {OLLAMA_MODEL}\n"
                "Then try again.]")
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    try:
        res = _ollama_request("/api/chat", payload)
        # expected: {"message":{"role":"assistant","content":"..."}}
        msg = res.get("message",{}).get("content","").strip()
        return msg or "[llm_empty_response]"
    except Exception as e:
        return f"[llm_error:{e}]"

@register_tool("llm.chat", help="Chat with the local Ollama model.",
    args={"prompt":("str","..."), "system":("str","You are a helpful assistant."), "model":("str",""), "temperature":("float",0.2), "max_tokens":("int",512)},
    tags=("ai","chat"))
def _tool_llm_chat(prompt: str, system: str="You are a helpful assistant.", model: str="", temperature: float=0.2, max_tokens: int=512) -> str:
    msgs = [
        {"role":"system","content":system},
        {"role":"user","content":prompt},
    ]
    return llm_chat(msgs, model or None, temperature, max_tokens)

# ===================== NL Command Router =====================
_GREET_RE = re.compile(r"^\s*(hi|hey+|hello|yo|sup|hiya|heyyy?)\b", re.I)
_ONE_WORD_CONFUSION = {"what","wut","wat","huh","ok","okay","k","hmm","hmmm","h","?","??","???"}

def _looks_like_tools_query(tl: str) -> bool:
    if re.search(r"\b(what|which|show|list|got|available|have|can you do)\b", tl) and re.search(r"\b(tools?|commands?|capabilities)\b", tl):
        return True
    if tl.strip() in {"tools","tool","commands","capabilities","what can you do"}:
        return True
    if re.search(r"\b(show|list)\b.*\btools?\b", tl):
        return True
    return False

def _tokens(s: str) -> List[str]:
    try: return shlex.split(s)
    except Exception: return s.split()

def _split_kv_and_pos(tokens: List[str]) -> Tuple[List[str], Dict[str,str]]:
    pos, kv = [], {}
    for tok in tokens:
        if "=" in tok and not tok.strip().startswith(("http://","https://")):
            k, v = tok.split("=",1); kv[k]=v
        else:
            pos.append(tok)
    return pos, kv

def _invoke_with_smart_defaults(name: str, rest_tokens: List[str]) -> str:
    spec = TOOLS[name]
    pos, kv = _split_kv_and_pos(rest_tokens)
    if len(pos) == 1 and len(spec.args) == 1 and not kv:
        return run_tool_std(name, pos_args=[pos[0]], kv_args={})
    if not pos and not kv:
        return run_tool_std(name, pos_args=[], kv_args={})
    return run_tool_std(name, pos_args=pos, kv_args=kv)

def _once_hint() -> str:
    global _HINT_SHOWN
    hint = ("Try: 'search the web for pandas dataframe tutorial', "
            "'file.list ~', 'pass.gen 24', or just ask me anything.")
    if _HINT_SHOWN: return "How can I help?"
    _HINT_SHOWN = True; return hint

def _route_nl_to_tool(tl: str) -> Optional[str]:
    # --- Web search intents ---
    m = re.search(r'\b(search|look\s*up|google|find)\b\s+(?:the\s+web\s+for\s+|for\s+)?(.+)', tl)
    if m:
        q = m.group(2).strip()
        if q and not q.startswith(("tool ","do ","run ","use ")):
            return run_tool_std("web.search", kv_args={"query": q})

    # --- Password gen ---
    if re.search(r'\b(generate|make|create)\b.*\bpassword\b', tl):
        m = re.search(r'(\d{2,3})', tl)
        length = m.group(1) if m else "16"
        return run_tool_std("pass.gen", kv_args={"length": length})

    # --- File list / read / write ---
    if re.search(r'\b(list|show)\b.*\b(dir|directory|folder|files?)\b', tl):
        m = re.search(r'\b(in|at|of)\s+(\S+)', tl)
        path = m.group(2) if m else "."
        return run_tool_std("file.list", kv_args={"path": path})
    if re.search(r'\b(read|open|view)\b.*\bfile\b', tl):
        m = re.search(r'\b(?:read|open|view)\b\s+(\S+)', tl)
        if m: return run_tool_std("file.read", kv_args={"path": m.group(1)})
    if re.search(r'\b(write|overwrite|save)\b.*\bfile\b', tl):
        m = re.search(r'\b(write|save)\b\s+(\S+)\s+(?:with|as)\s+(.+)$', tl)
        if m: return run_tool_std("file.write", kv_args={"path": m.group(2), "text": m.group(3)})
    if re.search(r'\b(create)\b.*\bfile\b', tl):
        m = re.search(r'\bcreate\b\s+(\S+)(?:\s+with\s+(.+))?$', tl)
        if m: return run_tool_std("file.create", kv_args={"path": m.group(1), "text": m.group(2) or ""})
    if re.search(r'\b(append|insert|replace)\b.*\bfile\b', tl):
        # rough: append/insert/replace {text} in {path}
        m = re.search(r'\b(append|insert|replace)\b\s+(.+?)\s+\b(in|into)\b\s+(\S+)(?:\s+at\s+line\s+(\d+))?', tl)
        if m:
            mode, text, _, path, line = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            kv = {"path": path, "mode": mode, "text": text}
            if line and mode == "insert": kv["line"] = line
            if mode == "replace": kv["pattern"] = text  # allow "replace PATTERN in FILE"
            return run_tool_std("file.edit", kv_args=kv)

    # --- WiFi / USB ---
    if re.search(r'\b(wifi|wi[- ]?fi)\b.*\b(scan|networks?)\b', tl):
        return run_tool_std("wifi.scan")
    if re.search(r'\b(usb)\b.*\b(list|devices?)\b', tl):
        return run_tool_std("usb.list")

    # --- Calculator ---
    cm = re.search(r'(?:calculate|compute|what is)\s+([0-9+\-*/().\s]+)$', tl)
    if cm:
        return run_tool_std("demo.calc", kv_args={"expr": cm.group(1).strip()})

    return None  # let LLM handle

def respond_to_user(user_input: str) -> str:
    text = user_input.strip()
    tl = text.lower().strip()

    # Greetings / small talk (short hint once)
    if _GREET_RE.match(text): return "Hey! " + _once_hint()
    if tl in _ONE_WORD_CONFUSION: return "Yep. " + _once_hint()

    # Tool listings / help
    if tl in {'help','h','/?','-h','--help','tools','tool','commands','capabilities'}:
        return tools_list()
    if _looks_like_tools_query(tl): return tools_list()
    if tl.startswith("help "):
        name = tl.split(None,1)[1].replace(" ",".")
        return tools_help(name)

    # Explicit tool calls
    if tl.startswith("tool "):
        try:
            name, pos, kv = parse_tool_invocation(tl)
            if name in TOOLS: return run_tool_std(name, pos_args=pos, kv_args=kv)
            similar = [n for n in TOOLS if name in n or n in name]
            return f"[tool_error: unknown tool '{name}'.{' Did you mean: ' + ', '.join(sorted(similar)[:3]) if similar else ''}]"
        except Exception as e:
            return f"[parse_error:{e}]"

    m = re.match(r'^\s*(do|run|use)(?:\s+tool)?\s+([a-z0-9_.-]+)(?:\s+(.*))?$', tl)
    if m:
        tool_name = m.group(2); rest = _tokens(m.group(3) or "")
        if tool_name in TOOLS:
            return _invoke_with_smart_defaults(tool_name, rest)
        if "." not in tool_name and any(tool_name in n.replace("."," ") for n in TOOLS):
            cands = [n for n in TOOLS if tool_name in n.replace("."," ")]
            if len(cands)==1: return _invoke_with_smart_defaults(cands[0], rest)
            return f"[tool_error: unknown tool '{tool_name}'. Did you mean: {', '.join(sorted(cands)[:3])}?]"
        similar = [n for n in TOOLS if tool_name in n or n in tool_name]
        return f"[tool_error: unknown tool '{tool_name}'.{' Did you mean: ' + ', '.join(sorted(similar)[:3]) if similar else ''}]"

    # Direct "<toolname> args..."
    m = re.match(r'^\s*([a-z0-9_.-]+)(?:\s+(.*))?$', tl)
    if m and m.group(1) in TOOLS:
        name = m.group(1); rest = _tokens(m.group(2) or "")
        return _invoke_with_smart_defaults(name, rest)

    # Natural-language routing to tools
    routed = _route_nl_to_tool(tl)
    if routed is not None:
        return routed

    # Fallback to LLM chat (poems, reasoning, general Q&A)
    system_prompt = (
        "You are Anacharssis_Z3, a helpful, concise assistant. "
        "Think carefully, keep answers clear. Write code and poetry when asked. "
        "Prefer factual accuracy; if unknown, say so briefly."
    )
    msgs = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":text},
    ]
    return llm_chat(msgs, model=OLLAMA_MODEL, temperature=0.3, max_tokens=700)

# ===================== Main REPL =====================
def main():
    print("Anacharssis_Z3 — Tools Framework (owner: Rostyslav)")
    print("Powered by Ollama + tools")
    print("Type 'tools' to list tools, or just ask me anything.")
    while True:
        try:
            user_input = input("> ")
            if not user_input: continue
            tl = user_input.strip().lower()
            if tl in {'quit','exit',':q','/quit'}:
                print("Goodbye!"); break
            resp = respond_to_user(user_input)
            print(resp)
        except KeyboardInterrupt:
            print("\nGoodbye!"); break
        except EOFError:
            print("\nGoodbye!"); break
        except Exception as e:
            print(f"[system_error: {e}]")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anacharssis_Z3 — Minimal AI Agent with Tools + LLM (Ollama)
Owner: Rostyslav
Version: Aug 2025 (Ollama chat, NL routing, working web.search)
"""

# ===================== Imports & Setup =====================
import os, re, shlex, json, math, ast, secrets, string, pathlib, subprocess, shutil, sys, time
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Tuple
from urllib import request, parse, error as urlerror
from html import unescape

HOME = pathlib.Path.home().resolve()
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5")
_HINT_SHOWN = False

def _norm(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = (pathlib.Path.cwd() / p).resolve()
    return p

# ===================== Tools Framework =====================
@dataclass
class ToolSpec:
    name: str
    fn: Callable
    help: str = ""
    args: Dict[str, Tuple[str, object]] = field(default_factory=dict)  # {name: (type, default|'...')}
    guard: Optional[Callable] = None
    tags: Tuple[str, ...] = ("general",)

TOOLS: Dict[str, ToolSpec] = {}

def register_tool(name: str, *, help: str, args: Dict[str, Tuple[str, object]] | None = None,
                  guard: Optional[Callable] = None, tags: Tuple[str, ...] = ("general",)):
    def deco(fn: Callable):
        TOOLS[name] = ToolSpec(name=name, fn=fn, help=help, args=args or {}, guard=guard, tags=tags)
        return fn
    return deco

_BOOL_TRUE = {"1","true","yes","on","y","ok"}

def _cast_arg(v: str, ty: str):
    if ty == "str": return v
    if ty == "int": return int(v)
    if ty == "float": return float(v)
    if ty == "bool": return v.strip().lower() in _BOOL_TRUE
    if ty == "path": return str(_norm(v))
    if ty == "json":
        try: return json.loads(v)
        except Exception as e: raise ValueError(f"invalid json: {e}")
    raise ValueError(f"unknown type '{ty}'")

def _normalize_args(spec: ToolSpec, pos_args: List[str], kv_args: Dict[str,str]):
    out = {}
    ordered = list(spec.args.items())  # preserve declared order
    for i, (k,(ty,default)) in enumerate(ordered):
        if i < len(pos_args):
            out[k] = _cast_arg(pos_args[i], ty)
        elif k in kv_args:
            out[k] = _cast_arg(kv_args[k], ty)
        else:
            if default == "...":
                raise ValueError(f"missing required arg '{k}'")
            out[k] = default
    extras = set(kv_args) - set(spec.args)
    if extras:
        raise ValueError("unknown args: " + ", ".join(sorted(extras)))
    return out

def parse_tool_invocation(s: str) -> Tuple[str, List[str], Dict[str,str]]:
    # expects: "tool <name> [pos...] [k=v ...]"
    parts = shlex.split(s)
    if len(parts) < 2: raise ValueError("usage: tool <name> [args...]")
    name = parts[1]
    pos, kv = [], {}
    for tok in parts[2:]:
        if "=" in tok:
            k, v = tok.split("=",1); kv[k] = v
        else:
            pos.append(tok)
    return name, pos, kv

def run_tool_std(name: str, *, pos_args: List[str] | None = None, kv_args: Dict[str,str] | None = None) -> str:
    pos_args = pos_args or []; kv_args = kv_args or {}
    spec = TOOLS.get(name)
    if not spec: return f"[tool_error: unknown tool '{name}']"
    try:
        args = _normalize_args(spec, pos_args, kv_args)
    except Exception as e:
        return f"[arg_error:{e}]"
    if spec.guard:
        err = spec.guard(args)
        if err: return err
    try:
        return spec.fn(**args)
    except Exception as e:
        return f"[tool_runtime_error:{e}]"

def tools_list() -> str:
    if not TOOLS: return "(no tools registered)"
    groups: Dict[str,List[str]] = {}
    for t in TOOLS.values():
        for tag in t.tags:
            groups.setdefault(tag, []).append(t.name)
    lines = []
    for tag in sorted(groups):
        lines.append(f"[{tag}]")
        for n in sorted(groups[tag]): lines.append(f"  - {n}")
    lines.append("\nTry: tools help <name>  or  tool <name> key=val")
    return "\n".join(lines)

def tools_help(name: Optional[str] = None) -> str:
    if not name: return tools_list()
    spec = TOOLS.get(name)
    if not spec: return f"(no such tool: {name})"
    def fmt_default(d): return "required" if d == "..." else json.dumps(d)
    arglines = [f"  {k}: {ty}, default={fmt_default(d)}" for k,(ty,d) in spec.args.items()]
    tags = ", ".join(spec.tags) if spec.tags else "general"
    return f"{name} — {spec.help}\nTags: {tags}\nArgs:\n" + ("\n".join(arglines) if arglines else "  (none)")

# ===================== Tools Implementation =====================
# ---------- File tools ----------
def _home_guard(args):
    p_raw = args.get("path","")
    if not p_raw: return None
    try:
        r = _norm(p_raw).resolve()
        if not (HOME in r.parents or r == HOME):
            return f"[guard: path must be inside HOME: {HOME}]"
    except Exception:
        return "[guard: invalid path]"
    return None

@register_tool("file.read", help="Read a UTF-8 text file.",
    args={"path": ("path","...")}, guard=_home_guard, tags=("files",))
def _tool_file_read(path: str) -> str:
    try: return _norm(path).read_text(encoding="utf-8")
    except Exception as e: return f"[read_error:{e}]"

@register_tool("file.write", help="Write text to a file (overwrite).",
    args={"path": ("path","..."), "text": ("str","...")}, guard=_home_guard, tags=("files",))
def _tool_file_write(path: str, text: str) -> str:
    try:
        p = _norm(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return f"OK: wrote {len(text)} bytes to {p}"
    except Exception as e:
        return f"[write_error:{e}]"

@register_tool("file.create", help="Create a new file (fails if exists). Seed with text if given.",
    args={"path": ("path","..."), "text": ("str","")}, guard=_home_guard, tags=("files",))
def _tool_file_create(path: str, text: str = "") -> str:
    try:
        p = _norm(path)
        if p.exists(): return f"[create_error: file exists: {p}]"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding="utf-8")
        return f"OK: created {p} ({len(text)} bytes)"
    except Exception as e:
        return f"[create_error:{e}]"

@register_tool("file.edit", help="Edit file. Modes: append, overwrite, insert(line=1-based), replace(pattern=regex).",
    args={"path": ("path","..."), "mode": ("str","append"), "text": ("str",""), "line": ("int",0), "pattern": ("str","")},
    guard=_home_guard, tags=("files",))
def _tool_file_edit(path: str, mode: str="append", text: str="", line: int=0, pattern: str="") -> str:
    try:
        p = _norm(path)
        if not p.exists(): return f"[edit_error: file not found: {p}]"
        content = p.read_text(encoding="utf-8")
        m = mode.lower()
        if m == "overwrite":
            p.write_text(text, encoding="utf-8"); return f"OK: overwrote {p} ({len(text)} bytes)"
        if m == "append":
            newc = content + ("" if content.endswith("\n") else "\n") + text
            p.write_text(newc, encoding="utf-8"); return f"OK: appended {len(text)} bytes to {p}"
        if m == "insert":
            lines = content.splitlines(True)
            idx = max(0, min((line-1) if line>0 else len(lines), len(lines)))
            lines.insert(idx, text if text.endswith("\n") else text + "\n")
            p.write_text("".join(lines), encoding="utf-8"); return f"OK: inserted at line {idx+1} in {p}"
        if m == "replace":
            if not pattern: return "[edit_error: replace mode requires pattern=...]"
            newc, n = re.subn(pattern, text, content, flags=re.MULTILINE)
            p.write_text(newc, encoding="utf-8"); return f"OK: replaced {n} occurrence(s) in {p}"
        return f"[edit_error: unknown mode '{mode}']"
    except Exception as e:
        return f"[edit_error:{e}]"

@register_tool("file.search", help="Search for files matching a glob (e.g. **/*.py).",
    args={"pattern": ("str","...")}, tags=("files","search"))
def _tool_file_search(pattern: str) -> str:
    try:
        matches = set()
        for base in (pathlib.Path.cwd(), HOME):
            for m in base.glob(pattern): matches.add(str(m))
        if not matches: return "No files found."
        return "Found files:\n" + "\n".join(f"  - {r}" for r in sorted(matches))
    except Exception as e:
        return f"[search_error:{e}]"

@register_tool("file.list", help="List files in a directory.",
    args={"path": ("path",".")}, tags=("files",))
def _tool_file_list(path: str) -> str:
    try:
        p = _norm(path)
        if not p.exists(): return f"[error: path does not exist: {p}]"
        if not p.is_dir(): return f"[error: not a directory: {p}]"
        files, dirs = [], []
        for item in p.iterdir():
            (dirs if item.is_dir() else files).append(item.name)
        lines = [f"Contents of {p}:"]
        for d in sorted(dirs):  lines.append(f"  [DIR]  {d}/")
        for f in sorted(files): lines.append(f"  [FILE] {f}")
        return "\n".join(lines)
    except Exception as e:
        return f"[scan_error:{e}]"

# ---------- System / Network ----------
@register_tool("usb.list", help="List USB devices (lsusb).",
    args={}, tags=("hardware","usb"))
def _tool_usb_list() -> str:
    try:
        if shutil.which("lsusb") is None:
            return "[usb_error: 'lsusb' not found. Install usbutils or run on a system that has it.]"
        result = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return "Connected USB devices:\n" + result.stdout.strip()
        return "[error: No output from lsusb]"
    except Exception as e:
        return f"[usb_error:{e}]"

@register_tool("wifi.scan", help="Scan Wi-Fi (nmcli).",
    args={}, tags=("wifi","network"))
def _tool_wifi_scan() -> str:
    try:
        if shutil.which("nmcli") is None:
            return "[wifi_error: 'nmcli' not found. Install NetworkManager or run on a system that has it.]"
        result = subprocess.run(["nmcli","device","wifi","list"], capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            return "Available Wi-Fi networks:\n" + result.stdout.strip()
        return "[error: Wi-Fi scan failed. Is NetworkManager running?]"
    except Exception as e:
        return f"[wifi_error:{e}]"

# ---------- Utility / Demo ----------
@register_tool("demo.calc", help="Evaluate a basic math expression (digits + + - * / ( ) .).",
    args={"expr": ("str","...")}, tags=("math","demo"))
def _tool_demo_calc(expr: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expr): return "[calc_error: invalid characters]"
        node = ast.parse(expr, mode="eval"); code = compile(node, "<calc>", "eval")
        return f"{expr} = {eval(code, {'__builtins__': {}}, vars(math))}"
    except Exception as e:
        return f"[calc_error:{e}]"

@register_tool("pass.gen", help="Generate a secure random password.",
    args={"length": ("int", 16), "symbols": ("bool", True)}, tags=("security","password"))
def _tool_pass_gen(length: int = 16, symbols: bool = True) -> str:
    try:
        length = max(4, min(int(length), 128))
        chars = string.ascii_letters + string.digits
        if symbols: chars += "!@#$%^&*"
        return ''.join(secrets.choice(chars) for _ in range(length))
    except Exception as e:
        return f"[pass_gen_error:{e}]"

# ---------- Web Search (DuckDuckGo HTML, no API key) ----------
def _http_get(url: str, data: Optional[bytes]=None, timeout: int=12) -> str:
    req = request.Request(url, data=data, headers={"User-Agent":"Mozilla/5.0 (Z3Agent)"})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")

@register_tool("web.search", help="Search the web (DuckDuckGo). Returns top results.",
    args={"query": ("str","..."), "max": ("int", 5)}, tags=("web","search"))
def _tool_web_search(query: str, max: int = 5) -> str:
    try:
        q = parse.urlencode({"q": query})
        html = _http_get("https://duckduckgo.com/html/?" + q)
        # crude parse of results
        items = []
        for m in re.finditer(r'<a rel="nofollow" class="result__a" href="([^"]+)".*?>(.*?)</a>', html, re.I|re.S):
            href = unescape(m.group(1))
            title = re.sub("<.*?>","",unescape(m.group(2))).strip()
            if title and href:
                items.append((title, href))
            if len(items) >= max: break
        if not items:
            return f"No results for: {query}"
        out = [f"Top results for: {query}"]
        for i,(t,u) in enumerate(items,1):
            out.append(f"{i}. {t}\n   {u}")
        return "\n".join(out)
    except urlerror.URLError as e:
        return f"[web_error: network issue — {e}]"
    except Exception as e:
        return f"[web_error:{e}]"

# ---------- LLM (Ollama) ----------
def _ollama_request(path: str, payload: dict, timeout: int=60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    url = f"{OLLAMA_BASE}{path}"
    req = request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="ignore")
            # /api/chat streams by lines if stream=true. We send stream=false, so plain JSON.
            return json.loads(txt)
    except Exception as e:
        raise RuntimeError(str(e))

def _ollama_alive() -> bool:
    try:
        req = request.Request(f"{OLLAMA_BASE}/api/tags")
        req.add_header("Accept","application/json")
        with request.urlopen(req, timeout=4) as resp:
            _ = resp.read()
        return True
    except Exception:
        return False

def llm_chat(messages: List[dict], model: Optional[str]=None, temperature: float=0.2, max_tokens: int=512) -> str:
    if not _ollama_alive():
        return ("[llm_error: Ollama is not reachable at "
                f"{OLLAMA_BASE}. Start it:\n  $ ollama serve\n"
                f"and pull a model:\n  $ ollama pull {OLLAMA_MODEL}\n"
                "Then try again.]")
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "options": {"temperature": float(temperature)},
        "stream": False,
    }
    try:
        res = _ollama_request("/api/chat", payload)
        # expected: {"message":{"role":"assistant","content":"..."}}
        msg = res.get("message",{}).get("content","").strip()
        return msg or "[llm_empty_response]"
    except Exception as e:
        return f"[llm_error:{e}]"

@register_tool("llm.chat", help="Chat with the local Ollama model.",
    args={"prompt":("str","..."), "system":("str","You are a helpful assistant."), "model":("str",""), "temperature":("float",0.2), "max_tokens":("int",512)},
    tags=("ai","chat"))
def _tool_llm_chat(prompt: str, system: str="You are a helpful assistant.", model: str="", temperature: float=0.2, max_tokens: int=512) -> str:
    msgs = [
        {"role":"system","content":system},
        {"role":"user","content":prompt},
    ]
    return llm_chat(msgs, model or None, temperature, max_tokens)

# ===================== NL Command Router =====================
_GREET_RE = re.compile(r"^\s*(hi|hey+|hello|yo|sup|hiya|heyyy?)\b", re.I)
_ONE_WORD_CONFUSION = {"what","wut","wat","huh","ok","okay","k","hmm","hmmm","h","?","??","???"}

def _looks_like_tools_query(tl: str) -> bool:
    if re.search(r"\b(what|which|show|list|got|available|have|can you do)\b", tl) and re.search(r"\b(tools?|commands?|capabilities)\b", tl):
        return True
    if tl.strip() in {"tools","tool","commands","capabilities","what can you do"}:
        return True
    if re.search(r"\b(show|list)\b.*\btools?\b", tl):
        return True
    return False

def _tokens(s: str) -> List[str]:
    try: return shlex.split(s)
    except Exception: return s.split()

def _split_kv_and_pos(tokens: List[str]) -> Tuple[List[str], Dict[str,str]]:
    pos, kv = [], {}
    for tok in tokens:
        if "=" in tok and not tok.strip().startswith(("http://","https://")):
            k, v = tok.split("=",1); kv[k]=v
        else:
            pos.append(tok)
    return pos, kv

def _invoke_with_smart_defaults(name: str, rest_tokens: List[str]) -> str:
    spec = TOOLS[name]
    pos, kv = _split_kv_and_pos(rest_tokens)
    if len(pos) == 1 and len(spec.args) == 1 and not kv:
        return run_tool_std(name, pos_args=[pos[0]], kv_args={})
    if not pos and not kv:
        return run_tool_std(name, pos_args=[], kv_args={})
    return run_tool_std(name, pos_args=pos, kv_args=kv)

def _once_hint() -> str:
    global _HINT_SHOWN
    hint = ("Try: 'search the web for pandas dataframe tutorial', "
            "'file.list ~', 'pass.gen 24', or just ask me anything.")
    if _HINT_SHOWN: return "How can I help?"
    _HINT_SHOWN = True; return hint

def _route_nl_to_tool(tl: str) -> Optional[str]:
    # --- Web search intents ---
    m = re.search(r'\b(search|look\s*up|google|find)\b\s+(?:the\s+web\s+for\s+|for\s+)?(.+)', tl)
    if m:
        q = m.group(2).strip()
        if q and not q.startswith(("tool ","do ","run ","use ")):
            return run_tool_std("web.search", kv_args={"query": q})

    # --- Password gen ---
    if re.search(r'\b(generate|make|create)\b.*\bpassword\b', tl):
        m = re.search(r'(\d{2,3})', tl)
        length = m.group(1) if m else "16"
        return run_tool_std("pass.gen", kv_args={"length": length})

    # --- File list / read / write ---
    if re.search(r'\b(list|show)\b.*\b(dir|directory|folder|files?)\b', tl):
        m = re.search(r'\b(in|at|of)\s+(\S+)', tl)
        path = m.group(2) if m else "."
        return run_tool_std("file.list", kv_args={"path": path})
    if re.search(r'\b(read|open|view)\b.*\bfile\b', tl):
        m = re.search(r'\b(?:read|open|view)\b\s+(\S+)', tl)
        if m: return run_tool_std("file.read", kv_args={"path": m.group(1)})
    if re.search(r'\b(write|overwrite|save)\b.*\bfile\b', tl):
        m = re.search(r'\b(write|save)\b\s+(\S+)\s+(?:with|as)\s+(.+)$', tl)
        if m: return run_tool_std("file.write", kv_args={"path": m.group(2), "text": m.group(3)})
    if re.search(r'\b(create)\b.*\bfile\b', tl):
        m = re.search(r'\bcreate\b\s+(\S+)(?:\s+with\s+(.+))?$', tl)
        if m: return run_tool_std("file.create", kv_args={"path": m.group(1), "text": m.group(2) or ""})
    if re.search(r'\b(append|insert|replace)\b.*\bfile\b', tl):
        # rough: append/insert/replace {text} in {path}
        m = re.search(r'\b(append|insert|replace)\b\s+(.+?)\s+\b(in|into)\b\s+(\S+)(?:\s+at\s+line\s+(\d+))?', tl)
        if m:
            mode, text, _, path, line = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            kv = {"path": path, "mode": mode, "text": text}
            if line and mode == "insert": kv["line"] = line
            if mode == "replace": kv["pattern"] = text  # allow "replace PATTERN in FILE"
            return run_tool_std("file.edit", kv_args=kv)

    # --- WiFi / USB ---
    if re.search(r'\b(wifi|wi[- ]?fi)\b.*\b(scan|networks?)\b', tl):
        return run_tool_std("wifi.scan")
    if re.search(r'\b(usb)\b.*\b(list|devices?)\b', tl):
        return run_tool_std("usb.list")

    # --- Calculator ---
    cm = re.search(r'(?:calculate|compute|what is)\s+([0-9+\-*/().\s]+)$', tl)
    if cm:
        return run_tool_std("demo.calc", kv_args={"expr": cm.group(1).strip()})

    return None  # let LLM handle

def respond_to_user(user_input: str) -> str:
    text = user_input.strip()
    tl = text.lower().strip()

    # Greetings / small talk (short hint once)
    if _GREET_RE.match(text): return "Hey! " + _once_hint()
    if tl in _ONE_WORD_CONFUSION: return "Yep. " + _once_hint()

    # Tool listings / help
    if tl in {'help','h','/?','-h','--help','tools','tool','commands','capabilities'}:
        return tools_list()
    if _looks_like_tools_query(tl): return tools_list()
    if tl.startswith("help "):
        name = tl.split(None,1)[1].replace(" ",".")
        return tools_help(name)

    # Explicit tool calls
    if tl.startswith("tool "):
        try:
            name, pos, kv = parse_tool_invocation(tl)
            if name in TOOLS: return run_tool_std(name, pos_args=pos, kv_args=kv)
            similar = [n for n in TOOLS if name in n or n in name]
            return f"[tool_error: unknown tool '{name}'.{' Did you mean: ' + ', '.join(sorted(similar)[:3]) if similar else ''}]"
        except Exception as e:
            return f"[parse_error:{e}]"

    m = re.match(r'^\s*(do|run|use)(?:\s+tool)?\s+([a-z0-9_.-]+)(?:\s+(.*))?$', tl)
    if m:
        tool_name = m.group(2); rest = _tokens(m.group(3) or "")
        if tool_name in TOOLS:
            return _invoke_with_smart_defaults(tool_name, rest)
        if "." not in tool_name and any(tool_name in n.replace("."," ") for n in TOOLS):
            cands = [n for n in TOOLS if tool_name in n.replace("."," ")]
            if len(cands)==1: return _invoke_with_smart_defaults(cands[0], rest)
            return f"[tool_error: unknown tool '{tool_name}'. Did you mean: {', '.join(sorted(cands)[:3])}?]"
        similar = [n for n in TOOLS if tool_name in n or n in tool_name]
        return f"[tool_error: unknown tool '{tool_name}'.{' Did you mean: ' + ', '.join(sorted(similar)[:3]) if similar else ''}]"

    # Direct "<toolname> args..."
    m = re.match(r'^\s*([a-z0-9_.-]+)(?:\s+(.*))?$', tl)
    if m and m.group(1) in TOOLS:
        name = m.group(1); rest = _tokens(m.group(2) or "")
        return _invoke_with_smart_defaults(name, rest)

    # Natural-language routing to tools
    routed = _route_nl_to_tool(tl)
    if routed is not None:
        return routed

    # Fallback to LLM chat (poems, reasoning, general Q&A)
    system_prompt = (
        "You are Anacharssis_Z3, a helpful, concise assistant. "
        "Think carefully, keep answers clear. Write code and poetry when asked. "
        "Prefer factual accuracy; if unknown, say so briefly."
    )
    msgs = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":text},
    ]
    return llm_chat(msgs, model=OLLAMA_MODEL, temperature=0.3, max_tokens=700)

# ===================== Main REPL =====================
def main():
    print("Anacharssis_Z3 — Tools Framework (owner: Rostyslav)")
    print("Powered by Ollama + tools")
    print("Type 'tools' to list tools, or just ask me anything.")
    while True:
        try:
            user_input = input("> ")
            if not user_input: continue
            tl = user_input.strip().lower()
            if tl in {'quit','exit',':q','/quit'}:
                print("Goodbye!"); break
            resp = respond_to_user(user_input)
            print(resp)
        except KeyboardInterrupt:
            print("\nGoodbye!"); break
        except EOFError:
            print("\nGoodbye!"); break
        except Exception as e:
            print(f"[system_error: {e}]")

if __name__ == "__main__":
    main()

