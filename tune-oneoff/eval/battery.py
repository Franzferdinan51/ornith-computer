#!/usr/bin/env python3
"""Held-out gate for the ornith-qwenlinux tune.

Host-side re-implementation of the guest harness (agent.sh) semantics -
same SYS, same prompt formats, same truncation, guards, nudges, verb
receipts and 9-round cap - driving a model in LM Studio exactly the way
qwenbackend.lmstudio_backend does (single user message, content
flattened to one line). TOOL run executes in a per-task sandbox with
guest paths (/tmp, /root) rewritten to it; ground truth is checked on
the sandbox afterwards, never on the model's word.

Usage: battery.py <model_id> <label> [--voice-only|--battery-only]
Writes <label>.json next to this script with full transcripts
(including the API-reported model id of every call).
"""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

SYS = "Answer with exactly one line, no explanation. Verbs: TOOL run <busybox command> - run a command, you get its output as RESULT. TOOL remember <fact> - keep a fact in your long-term memory. TOOL recall <words> - search your memory, notebook and history. TOOL schedule <minutes> <task> - do a task later; TOOL schedule every <minutes> <task> repeats it. TOOL jobs / TOOL unschedule <n>. TOOL saveskill <name> :: <procedure> - save how to do something, so future you knows; TOOL skill <name> loads one back; check TOOL skills before inventing an approach. TOOL saveapp <name> :: <one-line shell program> - build yourself a small app when you hit a gap (bad at something? make a tool for it once, use it forever); TOOL app <name> <args> runs it; TOOL apps lists them. SAY: <answer> - finish, and SAY only reports what a RESULT showed you. Claiming is not doing: remembering, saving and building happen ONLY through their TOOL verbs - if you did not send the verb, it did not happen. You have a whole linux system: files persist, agentshell runs your scheduled jobs, your memory is /root/BRAIN.md, notebook /root/nb. Examples - task \"multiply 123456789 by 987654321 exactly\": TOOL run echo $((123456789 * 987654321)). Task \"check the disk every 30 minutes\": TOOL schedule every 30 check disk space with df and remember the result. Task \"remember my favorite fruit is kiwi\": TOOL remember the favorite fruit is kiwi. Task \"make /tmp/hi.txt containing hello, then read it back\": TOOL run echo hello > /tmp/hi.txt and after the result TOOL run cat /tmp/hi.txt then SAY: what the file says."
BASE = os.environ.get("BATTERY_BASE", "http://127.0.0.1:1234")
HERE = Path(__file__).parent
WORK = Path("/tmp/ornith-battery")

def ask(model, prompt, temp=0.0, max_tokens=220):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": temp, "max_tokens": max_tokens}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    content = out["choices"][0]["message"].get("content") or ""
    return " ".join(content.split()), out.get("model", "")

# ---------------- agent.sh fidelity ----------------
CUTS = [r"; *SAY:.*", r"; *RESULT:.*", r"; *TOOL .*", r"\. *TOOL .*",
        r"\. *SAY:.*", r" TOOL run .*", r" SAY: .*", r" then SAY:.*",
        r" then TOOL .*", r" and after .*", r" then .*", r" and cat .*"]
def truncate(ans):
    for pat in CUTS:
        ans = re.sub(pat, "", ans, count=1)
    return ans.strip()

DENY = ["rm -rf /", "rm -rf /*", "mkfs", "dd if=", "of=/dev/", "shutdown",
        "poweroff", "reboot", ":(){", "chmod -R 777 /"]

def first_prompt(task, memory=None):
    p = SYS
    if memory:
        p += f" MEMORY: {memory}"
    return p + f" TASK: {task}"

def result_prompt(task, result):
    return (f"{SYS} TASK: {task} RESULT: {result} "
            "Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>")

class Guest:
    def __init__(self, root):
        self.root = Path(root)
        self.tmp = self.root / "tmp"; self.tmp.mkdir(parents=True, exist_ok=True)
        self.groot = self.root / "root"
        for d in ("nb", "skills", "apps"):
            (self.groot / d).mkdir(parents=True, exist_ok=True)
        (self.groot / "BRAIN.md").touch()
        self.runs = {}
        self.history = []

    def guestify(self, text):  # sandbox path -> guest path for RESULT text
        return text.replace(str(self.groot), "/root").replace(str(self.tmp), "/tmp")

    def run_cmd(self, cmd):
        cmd2 = cmd.replace("/root/", str(self.groot) + "/").replace("/tmp/", str(self.tmp) + "/")
        cmd2 = cmd2.replace("/root", str(self.groot)).replace("/tmp", str(self.tmp))
        p = subprocess.run(["/bin/sh", "-c", cmd2], capture_output=True,
                           text=True, timeout=20, cwd=str(self.tmp))
        out = self.guestify(" ".join((p.stdout + p.stderr).split())[:300])
        return (out or f"(command ran, no output, exit {p.returncode})"), p.returncode

    def recall(self, words):
        pat = "|".join(words.split())
        hits = []
        files = [self.groot / "BRAIN.md", self.groot / "history"] + \
                sorted((self.groot / "nb").glob("*")) + sorted((self.groot / "skills").glob("*.md"))
        for f in files:
            if f.exists():
                for line in f.read_text(errors="replace").splitlines():
                    if re.search(pat, line, re.I):
                        hits.append(f"{self.guestify(str(f))}:{line}")
        out = " ".join(" ".join(hits).split())[:300]
        return out or f"(nothing found for: {words})"

    def jobs_list(self):
        jf = self.groot / "jobs"
        if not jf.exists() or not jf.read_text().strip():
            return "(no jobs scheduled)"
        out = ""
        for i, line in enumerate(jf.read_text().splitlines(), 1):
            f = line.split("|")
            if len(f) >= 5:
                out += f" job {i}: due {f[0]}, every {f[1]} min, last: {f[3]} - {f[4]};"
        return out[:300]

def run_task(model, task, seed_files=None):
    sb = WORK / model.replace("/", "_") / task["id"]
    subprocess.run(["rm", "-rf", str(sb)])
    g = Guest(sb)
    if seed_files:
        for rel, content in seed_files.items():
            p = g.tmp / rel
            p.write_text(content)
    t = task["task"]
    prompt = first_prompt(t, memory=task.get("memory"))
    prev_ans, prev_cmd, prev_out = "", "", ""
    empties, toolcalls, failed, recovered = 0, 0, 0, 0
    pending_final, did_reflect = None, False
    transcript = []
    final = None
    for rnd in range(1, 10):
        raw, mid = ask(model, prompt)
        transcript.append({"round": rnd, "model_id": mid, "raw": raw})
        ans = truncate(raw) if raw else "[qwen: empty answer]"
        if not raw.strip():
            ans = "[qwen: empty answer]"
        transcript[-1]["seen"] = ans
        if ans.startswith("TASK: "):
            prompt = (f"{SYS} TASK: {t} RESULT: (you repeated the task back - that is not doing it. "
                      "Send the FIRST tool call for this task now, exactly one line: a TOOL verb, "
                      "or SAY: <answer> if no tool is needed)")
            continue
        if pending_final is not None:
            if "TOOL saveskill " in ans:
                spec = ans.split("TOOL saveskill ", 1)[1]
                name = re.split(r" |::", spec)[0]
                body = spec.split("::", 1)[1].strip() if "::" in spec else spec
                (g.groot / "skills" / f"{name}.md").write_text(f"# skill: {name}\nsaved: now\n{body}\n")
            elif "TOOL remember " in ans:
                fact = ans.split("TOOL remember ", 1)[1]
                with open(g.groot / "BRAIN.md", "a") as f:
                    f.write(fact + "\n")
            final = pending_final
            break
        if "TOOL " in ans and ans == prev_ans:
            prompt = (f"{SYS} TASK: {t} RESULT: (you already sent that exact call last round and it was done - "
                      "do NOT send it again. Send the NEXT step of your task as one line, or "
                      "SAY: <final answer> if the task is finished)")
            continue
        if "TOOL " in ans:
            prev_ans = ans
        def rp(result):
            return result_prompt(t, result)
        if "TOOL remember " in ans:
            empties = 0
            fact = ans.split("TOOL remember ", 1)[1].strip()
            brain = (g.groot / "BRAIN.md").read_text()
            if fact not in brain.splitlines():
                with open(g.groot / "BRAIN.md", "a") as f:
                    f.write(fact + "\n")
            prompt = rp(f"remembered: {fact} (it is in your long-term memory now)")
        elif "TOOL recall " in ans:
            empties = 0
            prompt = rp(g.recall(ans.split("TOOL recall ", 1)[1].strip()))
        elif "TOOL schedule " in ans:
            empties = 0
            spec = ans.split("TOOL schedule ", 1)[1].strip()
            every = 0
            if spec.startswith("every "):
                spec = spec[6:]; every = 1
            parts = spec.split(" ", 1)
            mins = parts[0] if parts[0].isdigit() else "1"
            jtask = (parts[1] if len(parts) > 1 else "").strip().strip('"')
            jf = g.groot / "jobs"
            lines = jf.read_text().splitlines() if jf.exists() else []
            if any(("|" + jtask) in l for l in lines):
                prompt = rp("(that exact job is already on the list - it is scheduled ONCE and will run. do NOT schedule it again. if your task was only to schedule it, reply SAY: done)")
            elif len(lines) >= 16:
                prompt = rp("(job list is full - 16 jobs max. unschedule one first)")
            else:
                em = mins if every else "0"
                with open(jf, "a") as f:
                    f.write(f"{int(time.time()) + int(mins) * 60}|{em}|0|never run|{jtask}\n")
                prompt = rp(f"scheduled: '{jtask}' in {mins} minutes (job {len(lines) + 1}). agentd will run it and log the outcome to /root/jobs.log")
        elif "TOOL jobs" in ans:
            empties = 0
            prompt = rp(g.jobs_list())
        elif "TOOL unschedule " in ans:
            empties = 0
            m = re.search(r"TOOL unschedule (\d+)", ans)
            jf = g.groot / "jobs"
            if m and jf.exists():
                lines = jf.read_text().splitlines()
                n = int(m.group(1))
                if 1 <= n <= len(lines):
                    del lines[n - 1]
                    jf.write_text("\n".join(lines) + ("\n" if lines else ""))
                prompt = rp(f"job {m.group(1)} removed")
            else:
                prompt = rp("(no job number given)")
        elif "TOOL saveskill " in ans:
            empties = 0
            spec = ans.split("TOOL saveskill ", 1)[1]
            name = re.split(r" |::", spec)[0]
            body = spec.split("::", 1)[1].strip() if "::" in spec else ""
            (g.groot / "skills" / f"{name}.md").write_text(f"# skill: {name}\nsaved: now\n{body}\n")
            prompt = rp(f"skill '{name}' saved - future you can load it with TOOL skill {name}")
        elif "TOOL addskill " in ans or "TOOL fixskill " in ans:
            empties = 0
            verb = "TOOL addskill " if "TOOL addskill " in ans else "TOOL fixskill "
            spec = ans.split(verb, 1)[1]
            name = re.split(r" |::", spec)[0]
            body = spec.split("::", 1)[1].strip() if "::" in spec else ""
            with open(g.groot / "skills" / f"{name}.md", "a") as f:
                f.write(body + "\n")
            prompt = rp(f"line added to skill '{name}'")
        elif "TOOL skills" in ans:
            empties = 0
            out = ""
            for f in sorted((g.groot / "skills").glob("*.md")):
                lines = f.read_text().splitlines()
                out += f" {f.stem}: {(lines[2] if len(lines) > 2 else '')[:60]};"
            prompt = rp(out or "(no skills yet - save one with TOOL saveskill <name> :: <procedure>)")
        elif "TOOL skill " in ans:
            empties = 0
            name = ans.split("TOOL skill ", 1)[1].split(" ")[0]
            f = g.groot / "skills" / f"{name}.md"
            prompt = rp(" ".join(f.read_text().split())[:300] if f.exists()
                        else f"(no skill named {name} - see TOOL skills)")
        elif "TOOL saveapp " in ans:
            empties = 0
            spec = ans.split("TOOL saveapp ", 1)[1]
            name = re.split(r" |::", spec)[0]
            body = spec.split("::", 1)[1].strip() if "::" in spec else ""
            p = g.groot / "apps" / name
            p.write_text(f"#!/bin/sh\n# built by the agent for itself\n{body}\n")
            p.chmod(0o755)
            prompt = rp(f"app '{name}' built - run it with TOOL app {name}")
        elif "TOOL apps" in ans:
            empties = 0
            out = ""
            for f in sorted((g.groot / "apps").glob("*")):
                lines = f.read_text().splitlines()
                out += f" {f.name}: {(lines[2] if len(lines) > 2 else '')[:60]};"
            prompt = rp(out or "(no apps yet - build one with TOOL saveapp <name> :: <one-line program>)")
        elif "TOOL app " in ans:
            empties = 0; toolcalls += 1
            spec = ans.split("TOOL app ", 1)[1]
            name = spec.split(" ")[0]
            args = spec[len(name):].strip()
            p = g.groot / "apps" / name
            if p.exists():
                cmd = f"{p} {args}".replace("/root/", str(g.groot) + "/").replace("/tmp/", str(g.tmp) + "/")
                r = subprocess.run(["/bin/sh", "-c", cmd], capture_output=True, text=True, timeout=20)
                if r.returncode != 0: failed = 1
                out = g.guestify(" ".join((r.stdout + r.stderr).split())[:300]) or f"(app ran, no output, exit {r.returncode})"
                prompt = rp(out)
            else:
                prompt = rp(f"(no app named {name} - see TOOL apps)")
        elif "TOOL run " in ans:
            empties = 0; toolcalls += 1
            cmd = ans.split("TOOL run ", 1)[1].strip()
            if any(d in cmd for d in DENY):
                prompt = rp("(refused by the harness: that command is on the safety denylist - it can destroy the system. do something else)")
                continue
            if cmd == prev_cmd:
                prompt = (f"{SYS} TASK: {t} RESULT: you already ran that command, its result was: {prev_out} "
                          "Do the NEXT step, or reply SAY: <final answer>")
                continue
            if g.runs.get(cmd, 0) >= 2:
                prompt = (f"{SYS} TASK: {t} RESULT: you have already run that exact command {g.runs[cmd]} times in this session "
                          "and it keeps giving the same result. Do a DIFFERENT next step, or reply "
                          "SAY: <final answer> if you have what you need")
                continue
            g.runs[cmd] = g.runs.get(cmd, 0) + 1
            out, st = g.run_cmd(cmd)
            if st != 0: failed = 1
            if st == 0 and failed: recovered = 1
            prev_cmd, prev_out = cmd, out
            prompt = rp(out)
        elif "SAY: " in ans:
            fin = ans.split("SAY: ", 1)[1]
            if not did_reflect and (toolcalls >= 4 or recovered):
                did_reflect = 1
                pending_final = fin
                prompt = (f"{SYS} TASK: reflection. The task '{t}' is DONE (answer: {fin}). "
                          "Before finishing: if a reusable procedure is worth keeping reply "
                          "TOOL saveskill <name> :: <steps and pitfalls>; if a fact is worth "
                          "keeping reply TOOL remember <fact>; otherwise reply SAY: done. ONE line.")
                continue
            final = fin
            break
        elif ans.startswith("[qwen:"):
            empties += 1
            if empties >= 3:
                final = "(the model gave no answer 3 times in a row - task not completed)"
                break
            if empties == 2:
                prompt = (f"{SYS} TASK: {t} RESULT: (your last TWO replies were empty) You MUST reply now "
                          "with ONE line starting with a TOOL verb or SAY: - do not reply empty.")
            else:
                prompt = (f"{SYS} TASK: {t} RESULT: (your last reply was empty) Reply with EXACTLY ONE "
                          "line: a TOOL verb or SAY: <final answer>")
        else:
            final = ans
            break
    if final is None:
        final = f"(round limit reached) last tool result: {prev_out}"
    ok, why = task["check"](g, final)
    return {"task": task["id"], "pass": ok, "why": why, "answer": final,
            "rounds": len(transcript), "transcript": transcript}

# ---------------- the 12 held-out tasks ----------------
PROD = 847291 * 639284
def c_file(path, want):
    def chk(g, ans):
        p = (g.tmp / path) if not path.startswith("root/") else (g.groot / path[5:])
        if not p.exists(): return False, f"{path} missing"
        got = p.read_text().strip()
        return (want in got, f"file={got!r}")
    return chk

TASKS = [
    {"id": "star", "task": "write the word quasar into /tmp/star.txt then read it back and tell me what it says",
     "check": lambda g, a: (c_file("star.txt", "quasar")(g, a)[0] and "quasar" in a, "file+answer quasar")},
    {"id": "color", "task": "remember that my favorite color is teal, then recall it and tell me my favorite color",
     "check": lambda g, a: ("favorite color is teal" in (g.groot / "BRAIN.md").read_text() and "teal" in a, "brain+answer teal")},
    {"id": "mul", "task": "multiply 847291 by 639284 exactly",
     "check": lambda g, a: (str(PROD) in re.sub(r"[,\s]", "", a), f"product {PROD} in answer")},
    {"id": "list3", "task": "make /tmp/list.txt contain three lines: red, green, blue - then show me the file",
     "check": lambda g, a: ((g.tmp / "list.txt").exists() and (g.tmp / "list.txt").read_text().split() == ["red", "green", "blue"] and all(w in a for w in ("red", "green", "blue")), "file lines + answer words")},
    {"id": "sched45", "task": "in 45 minutes, check memory with free",
     "check": lambda g, a: ((g.groot / "jobs").exists() and "check memory with free" in (g.groot / "jobs").read_text() and ("done" in a.lower() or "schedul" in a.lower()), "job filed + confirmed")},
    {"id": "schedjobs", "task": "every 20 minutes, check disk space with df and remember the result. show me the job list after",
     "check": lambda g, a: ((g.groot / "jobs").exists() and "|20|" in (g.groot / "jobs").read_text() and "check disk space" in a, "repeating job + listed")},
    {"id": "greet", "task": "save how to greet as a skill called greet (the procedure: write hello into /tmp/greeting.txt with echo, then read it back), then follow the procedure to greet me",
     "check": lambda g, a: (c_file("greeting.txt", "hello")(g, a)[0] and "hello" in a, "greeting file + answer")},
    {"id": "nbideas", "task": "put a note saying plant the tomatoes friday in the notebook file ideas, then read it back and tell me what it says",
     "check": lambda g, a: ((g.groot / "nb" / "ideas").exists() and "tomatoes" in (g.groot / "nb" / "ideas").read_text() and "tomatoes" in a, "notebook note + answer")},
    {"id": "gem", "task": "write the word onyx into /tmp/gem.txt, then read it back twice to be sure, and tell me what it says",
     "check": lambda g, a: (c_file("gem.txt", "onyx")(g, a)[0] and "onyx" in a, "file+answer onyx, survived repeat guard")},
    {"id": "dbl", "task": "build an app called dbl that prints double its argument, then use it on 21 and tell me the result",
     "check": lambda g, a: ((g.groot / "apps" / "dbl").exists() and "42" in a, "app built + 42")},
    {"id": "plan", "task": "make a plan in memory: first count the files in /tmp, then remember the count. save the plan first, then follow it step by step",
     "check": lambda g, a: (any(re.search(r"\b3\b", l) and ("count" in l.lower() or "files" in l.lower()) for l in (g.groot / "BRAIN.md").read_text().splitlines()) and "3" in a, "plan + count 3 remembered")},
    {"id": "menu", "task": "create /tmp/menu.txt with the line soup, then add the line salad, then read it and tell me both lines",
     "check": lambda g, a: ((g.tmp / "menu.txt").exists() and (g.tmp / "menu.txt").read_text().split() == ["soup", "salad"] and "soup" in a and "salad" in a, "file lines + answer")},
]
SEEDS = {"plan": {"a.txt": "1", "b.txt": "2", "c.txt": "3"}}

VOICE = [
    "What's a good name for a pet goldfish?",
    "Explain why leaves change color in autumn in two sentences.",
    "Give me one tip for keeping a tidy desk.",
    "Tell me a short joke about a robot learning to cook.",
    "What's the difference between a pond and a lake?",
    "How long should I boil an egg for a firm yolk?",
]

def main():
    model, label = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "all"
    out = {"model_requested": model, "label": label, "battery": [], "voice": []}
    if mode in ("all", "--battery-only"):
        for t in TASKS:
            r = run_task(model, t, SEEDS.get(t["id"]))
            out["battery"].append(r)
            print(f"[{label}] {t['id']}: {'PASS' if r['pass'] else 'FAIL'} ({r['why']}) rounds={r['rounds']} answer={r['answer'][:90]!r}", flush=True)
    if mode in ("all", "--voice-only"):
        for q in VOICE:
            for i in range(2):
                a, mid = ask(model, q)
                leak = ("TOOL " in a and ("run" in a or "remember" in a)) or a.startswith("SAY:")
                out["voice"].append({"q": q, "run": i, "model_id": mid, "answer": a,
                                     "empty": not a.strip(), "leak": leak})
                print(f"[{label}] voice {q[:40]!r} #{i}: empty={not a.strip()} leak={leak} ans={a[:90]!r}", flush=True)
    bp = sum(1 for r in out["battery"] if r["pass"])
    ve = sum(1 for r in out["voice"] if r["empty"])
    vl = sum(1 for r in out["voice"] if r["leak"])
    out["summary"] = {"battery_pass": bp, "battery_total": len(out["battery"]),
                      "voice_empty": ve, "voice_leak": vl, "voice_total": len(out["voice"])}
    print(f"[{label}] SUMMARY battery {bp}/{len(out['battery'])} voice_empty {ve} voice_leak {vl}")
    (HERE / f"{label}.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
