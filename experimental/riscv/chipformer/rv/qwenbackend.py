"""Host backends for the Qwen coprocessor device (linux.py, QWEN_BASE).

A backend is a plain callable: prompt string in, answer string out.
The device is deliberately backend-blind — today these are:

  * ``echo``     — deterministic, offline; proves the plumbing end to end.
  * ``lmstudio`` — whatever model is *already loaded* in LM Studio
                   (auto-detected via /v1/models; never triggers a load).
  * ``builtin``  — the preconfigured seat for the harness: **Ornith 1.5 9B**,
                   the brain this fork is configured for (the guest runs on
                   the fast core beside it until the Ornith fusion lands),
                   resolved by family from LM Studio's model list and
                   pinned (a pristine/base build wins over fine-tunes
                   when several match). This is the harness's default:
                   hermes-agent's brain is the built-in weights, not
                   whatever happens to be loaded. ``ORNITH_MODEL`` pins an
                   exact model id instead, and ``ORNITH_FAMILY`` overrides
                   the family substring — the two knobs that keep a
                   fused/GGUF build seated under its new name.

The endgame backend is the fused Ornith checkpoint's own language layers
(the same model whose chip layers will run the guest CPU after fusion);
it drops in behind this same callable when the in-weights Linux run
lands, with no guest or device change. See docs/linux.md.
"""

import json
import os
import urllib.request


def echo_backend(prompt: str) -> str:
    return f"qwen-echo: {prompt}"


def lmstudio_backend(base_url: str | None = None, proxy: str | None = None,
                     model: str | None = None, temperature=None,
                     max_tokens=None):
    """Return a backend that asks a model in LM Studio.

    base_url: LM Studio root (default env ORNITH_LM_URL, else
    http://127.0.0.1:1234 — right when the runner sits next to LM Studio,
    e.g. on the Mac). proxy: explicit HTTP proxy (env ORNITH_LM_PROXY); from
    the VM the tailnet route is the :3130 variant of HTTPS_PROXY.
    model: pin an exact model id (or env ORNITH_MODEL); without a pin the
    first listed model is used (the original auto-detect behavior).
    A pin of "auto" (the dashboard's word) also means unpinned.
    temperature/max_tokens: explicit values win; else ORNITH_TEMP /
    ORNITH_MAX_TOKENS; else 0.7 / 220.
    """
    base = (base_url or os.environ.get("ORNITH_LM_URL")
            or "http://127.0.0.1:1234").rstrip("/")
    proxy = proxy or os.environ.get("ORNITH_LM_PROXY")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy else urllib.request.ProxyHandler({}))

    def call(path: str, payload: dict | None = None, timeout: float = 30) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            raise RuntimeError(f"LM Studio request to {base}{path} failed: {e}")

    ids = [m.get("id") for m in (call("/v1/models", timeout=15).get("data") or [])
           if m.get("id")]
    if not ids:
        raise RuntimeError(f"no model loaded in LM Studio at {base}")
    pin = model or os.environ.get("ORNITH_MODEL")
    if pin == "auto":  # the dashboard's word for "unpinned", not a model id
        pin = None
    model = pin or ids[0]
    if model not in ids:
        raise RuntimeError(f"pinned model {model!r} is not available in "
                           f"LM Studio at {base} (have: {', '.join(ids)})")
    try:
        temperature = float(temperature if temperature is not None
                            else os.environ.get("ORNITH_TEMP", "0.7"))
        max_tokens = int(max_tokens if max_tokens is not None
                         else os.environ.get("ORNITH_MAX_TOKENS", "220"))
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            f"bad temperature/max_tokens (arg or "
            f"ORNITH_TEMP/ORNITH_MAX_TOKENS): {e}")

    def ask(prompt: str) -> str:
        out = call("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, timeout=600)
        try:
            content = out["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise RuntimeError(f"unexpected chat response from {model!r} at "
                               f"{base}: {str(out)[:200]!r}")
        return content.strip() or "[qwen: empty answer]"

    ask.model = model  # for the runner's log line
    return ask


# The built-in weights: Ornith 1.5 9B, the seat model of this fork —
# the harness is preconfigured to think with it. Broad on purpose: the
# stock id, a fused build, and a GGUF all carry the substring.
BUILTIN_FAMILY = "ornith"
# ...but when several Ornith builds are loaded, the 9B this fork is
# configured for outranks its bigger siblings; they still seat fine
# when it is absent.
PREFERRED_SEAT = "ornith-1.5-9b"


def _rig_settings():
    """The dashboard's settings file when this runs on a rig ({} when
    absent or unparseable). Lets the Settings UI drive the guest brain
    too: explicit args win, then env, then these, then code defaults."""
    path = os.environ.get("ORNITH_DASH_SETTINGS") or os.path.expanduser(
        "~/ornith-test/dash-settings.json")
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def pick_builtin_model(ids, pin=None, family=BUILTIN_FAMILY,
                       preferred=PREFERRED_SEAT):
    """Pick the seated model id from LM Studio's model list (pure: no I/O).

    An exact pin wins when it names a listed id; otherwise the family
    substring decides (the preferred seat outranks bigger Ornith
    siblings, then a pristine/base build wins over tagged fine-tunes).
    Raises RuntimeError instead of
    serving a different model — the harness must think with the
    built-in weights or say so. After a fuse/GGUF conversion the new
    build keeps working here as long as its LM Studio id still carries
    the family substring (e.g. an ornith-fused-q8 GGUF does); under
    a wholly new name, seat it with ORNITH_MODEL/ORNITH_FAMILY.
    """
    ids = [i for i in ids if i]
    if pin and pin != "auto":
        if pin not in ids:
            raise RuntimeError(
                f'pinned model {pin!r} is not loaded '
                f'(have: {", ".join(ids) or "no models"})')
        return pin
    family = (family or BUILTIN_FAMILY).lower()
    preferred = (preferred or PREFERRED_SEAT).lower()
    fam = [i for i in ids if family in i.lower()]
    if not fam:
        raise RuntimeError(
            f"the built-in weights ({family}) are not available in "
            f"LM Studio (have: {', '.join(ids) or 'no models'}); load a "
            f"matching build first, or point ORNITH_MODEL/ORNITH_FAMILY at "
            f"the fused build's id")
    # the preferred 9B seat outranks bigger Ornith siblings; then the
    # pristine/base build (fewest dash-separated tags after the org
    # prefix) wins over tagged fine-tunes
    fam.sort(key=lambda i: (
        0 if preferred in i.lower() else 1,
        i.lower().rsplit("/", 1)[-1].count("-"), i))
    return fam[0]


def builtin_backend(base_url: str | None = None, proxy: str | None = None):
    """The harness's preconfigured brain: Ornith 1.5 9B in LM Studio.

    Resolution order is explicit args, then env, then the dashboard's
    Settings file on a rig, then code defaults — so the Settings UI
    drives the guest brain too, while env stays the operator override.
    Pins an exact id (ORNITH_MODEL, or the UI's model unless "auto");
    otherwise resolves the family (ORNITH_FAMILY, the UI's seat_family,
    or "ornith") and the preferred seat (ORNITH_PREFERRED, the UI's
    seat_preferred, or the 9B). Raises when nothing matches — the
    harness must think with the built-in weights or say so, never
    silently fall back to a different model. Generation params come
    from ORNITH_TEMP / ORNITH_MAX_TOKENS, the UI's guest_temperature /
    guest_max_tokens, or 0.7 / 1024.
    """
    rig = _rig_settings()
    base = (base_url or os.environ.get("ORNITH_LM_URL")
            or rig.get("backend_url")
            or "http://127.0.0.1:1234").rstrip("/")
    proxy = proxy or os.environ.get("ORNITH_LM_PROXY")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy else urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "/v1/models")
    try:
        with opener.open(req, timeout=15) as r:
            data = json.loads(r.read()).get("data") or []
    except Exception as e:
        raise RuntimeError(f"LM Studio request to {base}/v1/models failed: {e}")
    ids = [m.get("id") for m in data if m.get("id")]
    pin = os.environ.get("ORNITH_MODEL")
    if not pin:
        m = (rig.get("model") or "").strip()
        pin = m if m and m != "auto" else None
    try:
        choice = pick_builtin_model(
            ids, pin=pin,
            family=os.environ.get("ORNITH_FAMILY")
            or rig.get("seat_family") or BUILTIN_FAMILY,
            preferred=os.environ.get("ORNITH_PREFERRED")
            or rig.get("seat_preferred") or PREFERRED_SEAT)
    except RuntimeError as e:
        raise RuntimeError(f"{e} at {base}")
    try:
        temperature = float(os.environ.get(
            "ORNITH_TEMP", rig.get("guest_temperature", 0.7)))
        max_tokens = int(os.environ.get(
            "ORNITH_MAX_TOKENS", rig.get("guest_max_tokens", 1024)))
    except (ValueError, TypeError) as e:
        raise RuntimeError(
            f"bad ORNITH_TEMP/ORNITH_MAX_TOKENS or "
            f"guest_temperature/guest_max_tokens: {e}")
    return lmstudio_backend(base, proxy, model=choice,
                            temperature=temperature, max_tokens=max_tokens)


# A prompt from inside the guest harness always carries this phrase
# (agent.sh's baked-in system prompt) - the bridge uses it to tell a
# harness prompt from a terminal user's question.
AGENT_MARK = "TOOL run <command> - you will get"

_BRIDGE_HINT = (
    " [This question arrives from a terminal on a small linux machine "
    "that has its own agent, hermes-agent, which can run shell commands "
    "and use files inside that machine. If answering well needs that "
    "work done, reply with exactly one line: AGENT: <the task for "
    "hermes-agent>. You will then receive its report and can answer.]")


FETCH_MARK = "FETCHURL "
FETCH_MAX_BYTES = 4096


def fetch_page(url: str) -> str:
    """Fetch a page for the guest agent (its TOOL fetch verb).

    The guest has no network, so the host fetches on its behalf:
    http(s) only, no credentials in the URL, 15 s timeout, 4 KB cap,
    flattened to one console-safe line. Failures come back as a
    ``[fetch: ...]`` notice, never an exception, so the session
    survives a dead link.
    """
    url = (url or "").strip().split()[0] if (url or "").strip() else ""
    try:
        if not url or not url.lower().startswith(("http://", "https://")):
            return f"[fetch: refused (http(s) only): {url[:80]}]"
        if "@" in url.split("/", 3)[2]:
            return "[fetch: refused (credentials in URL)]"
        req = urllib.request.Request(
            url, headers={"User-Agent": "ornith-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read(FETCH_MAX_BYTES + 1)
        if len(raw) > FETCH_MAX_BYTES:
            raw = raw[:FETCH_MAX_BYTES]
        text = " ".join(raw.decode("utf-8", "replace").split())
        return text[:2000] or "[fetch: page was empty]"
    except Exception as e:
        return f"[fetch: failed: {str(e)[:160]}]"


def fetch_backend_wrapper(base):
    """Wrap a backend so ``FETCHURL <url>`` prompts are fetched by the
    host instead of asked of the model. The guest harness sends exactly
    that marker (TOOL fetch); everything else passes straight through.
    No guest or device change: the marker rides the existing QWEN:
    console protocol and needs no fastlinux.c change.
    """
    def ask(prompt: str) -> str:
        if isinstance(prompt, str) and prompt.startswith(FETCH_MARK):
            return fetch_page(prompt[len(FETCH_MARK):])
        return base(prompt)

    ask.model = getattr(base, "model", "fetch")
    return ask


def agent_bridge_backend(base):
    """Wrap a backend so the model in the seat can *use* the guest's
    hermes-agent: an ordinary question is offered the choice, and if the
    model replies ``AGENT: <task>`` the wrapper returns the
    ``[[RUN-AGENT: <task>]]`` marker instead. The guest-side ask program
    sees the marker, runs hermes-agent right there in the same Linux
    partition (same files), and asks again with a ``HERMES REPORT:``
    folded in, which this wrapper passes straight to the base model.
    Prompts from inside the harness itself always go straight to the
    base model - delegation is one level deep, never recursive.
    """
    def ask(prompt: str) -> str:
        if AGENT_MARK in prompt or "HERMES REPORT:" in prompt:
            return base(prompt)
        ans = str(base(prompt + _BRIDGE_HINT))
        flat = " ".join(ans.split())
        i = flat.find("AGENT:")
        if i >= 0:
            task = flat[i + 6:].strip()[:200]
            if task:
                return f"[[RUN-AGENT: {task}]]"
        return ans

    ask.model = getattr(base, "model", "agent-bridge")
    return ask


def make_backend(name: str, lm_url: str | None = None):
    if name == "echo":
        return echo_backend
    if name == "lmstudio":
        return lmstudio_backend(lm_url)
    if name == "builtin":
        return builtin_backend(lm_url)
    raise ValueError(f"unknown ornith backend {name!r}")
