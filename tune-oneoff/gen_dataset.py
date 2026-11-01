#!/usr/bin/env python3
"""Dataset builder for the Ornith 1.5 9B Qwen-Linux one-off tune.

Sources: the harness itself (agent.sh SYS + result/nudge prompt formats,
ask.sh one-line contract, the bare circuit agent's store/recall/mul
protocol from circuit-builtin.log). All rows synthesized; no private
data. Held-out battery strings (eval/battery.py) are checked against
every generated row - a row containing a battery marker is dropped.

Output: dataset/train.jsonl, dataset/valid.jsonl - {"messages": [...]}
"""
import json, random, re, sys
from pathlib import Path

HERE = Path(__file__).parent
SYS = json.loads((HERE / "sys.json").read_text())["SYS"]
rng = random.Random(20261031)

# ---------- held-out battery markers (must never appear in training) ----
BATTERY_MARKERS = [
    "quasar", "/tmp/star.txt", "teal", "847291", "639284", "/tmp/list.txt",
    "check memory with free in 45", "skill greet", "/root/nb/ideas",
    "app dbl", "banana", "/tmp/hello.txt", "/tmp/deep.txt",
    "123456 * 654321", "word under fruit",
]

def clean(rows):
    out = []
    for r in rows:
        blob = json.dumps(r)
        if any(m in blob for m in BATTERY_MARKERS):
            continue
        out.append(r)
    return out

# ---------- prompt builders (verbatim formats from agent.sh) ------------
def first_prompt(task, memory=None, soul=None):
    p = SYS
    if soul:
        p += f" YOU: {soul}"
    if memory:
        p += f" MEMORY: {memory}"
    return p + f" TASK: {task}"

def result_prompt(task, result):
    return (f"{SYS} TASK: {task} RESULT: {result} "
            "Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>")

def convo(turns):
    """turns: list of (user, assistant) pairs -> messages row"""
    msgs = []
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return {"messages": msgs}

# ---------- slot pools ----------------------------------------------------
WORDS = ["mango", "plum", "cedar", "onyx", "velvet", "harbor", "cobalt",
         "maple", "saffron", "willow", "granite", "meadow", "pepper",
         "sierra", "tundra", "violet", "walnut", "zephyr", "amber",
         "birch", "clover", "dune", "ember", "flint", "grove", "hazel",
         "iris", "juniper", "keel", "linen", "mosaic", "nettle", "opal",
         "prairie", "quartz", "rowan", "sorrel", "thistle", "umber"]
FILES = ["/tmp/note.txt", "/tmp/memo.txt", "/tmp/data.txt", "/tmp/log.txt",
         "/tmp/out.txt", "/tmp/report.txt", "/tmp/draft.txt", "/tmp/info.txt",
         "/tmp/jot.txt", "/tmp/entry.txt", "/tmp/record.txt", "/tmp/page.txt"]
NAMES = ["sam", "lee", "jo", "kim", "ash", "rue", "max", "ivy", "noa", "eli"]
FACTS = [
    ("the wifi password is written on the router label", "wifi password"),
    ("the backup runs on sundays", "backup day"),
    ("the greenhouse sensor is on channel 4", "greenhouse sensor channel"),
    ("the meeting moved to friday", "meeting day"),
    ("the spare key is under the blue pot", "spare key"),
    ("the server room code is taped inside the cabinet", "server room code"),
    ("the watering schedule is every other day", "watering schedule"),
    ("the project codename is lantern", "project codename"),
    ("the printer is on the second floor", "printer location"),
    ("the gate code changes monthly", "gate code"),
    ("the feed delivery arrives on tuesdays", "feed delivery day"),
    ("the workshop light switch is behind the door", "workshop light switch"),
]

def big_pair():
    return rng.randint(10000, 999999), rng.randint(10000, 999999)

rows = []

# ============ 1. agent protocol traces (full sessions) ===================
def t_write_read():
    w, f = rng.choice(WORDS), rng.choice(FILES)
    task = f"write the word {w} into {f} then read it back and tell me what it says"
    return convo([
        (first_prompt(task), f"TOOL run echo {w} > {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run cat {f}"),
        (result_prompt(task, w), f"SAY: the file says {w}"),
    ])

def t_write_lines():
    f = rng.choice(FILES)
    a, b, c = rng.sample(WORDS, 3)
    task = f"make {f} contain three lines: {a}, {b}, {c} - then show me the file"
    return convo([
        (first_prompt(task), f"TOOL run echo {a} > {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run echo {b} >> {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run echo {c} >> {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run cat {f}"),
        (result_prompt(task, f"{a} {b} {c}"),
         f"SAY: the file contains {a}, {b}, {c}"),
    ])

def t_arith():
    a, b = big_pair()
    op, sym, val = rng.choice([
        ("multiply", "*", a * b), ("add", "+", a + b),
        ("subtract", "-", a - b)])
    task = f"{op} {a} by {b} exactly" if op == "multiply" else \
           f"what is {a} {sym} {b}, exactly"
    return convo([
        (first_prompt(task), f"TOOL run echo $((${a} {sym} {b}))"),
        (result_prompt(task, str(val)), f"SAY: {a} {sym} {b} = {val}"),
    ])

def t_remember():
    fact, _ = rng.choice(FACTS)
    task = f"remember that {fact}"
    return convo([
        (first_prompt(task), f"TOOL remember {fact}"),
        (result_prompt(task, f"remembered: {fact} (it is in your long-term memory now)"),
         f"SAY: remembered - {fact}"),
    ])

def t_remember_recall():
    fact, key = rng.choice(FACTS)
    task = f"what do you remember about {key}?"
    mem = f"the user said: {fact}."
    return convo([
        (first_prompt(task, memory=mem), f"TOOL recall {key}"),
        (result_prompt(task, f"/root/BRAIN.md:{fact}"),
         f"SAY: {fact}"),
    ])

def t_recall_miss():
    w = rng.choice(WORDS)
    task = f"what did I tell you about the {w} shipment?"
    return convo([
        (first_prompt(task), f"TOOL recall {w} shipment"),
        (result_prompt(task, f"(nothing found for: {w} shipment)"),
         f"SAY: I have nothing stored about the {w} shipment"),
    ])

def t_schedule_once():
    n = rng.choice([10, 15, 20, 25, 40, 50, 90])
    job = rng.choice(["check disk space with df", "check memory with free",
                      "count the files in /tmp", "check the time with date"])
    task = f"in {n} minutes, {job}"
    due = 1762000000 + n * 60
    return convo([
        (first_prompt(task), f"TOOL schedule {n} {job}"),
        (result_prompt(task, f"scheduled: '{job}' in {n} minutes (job 1). agentd will run it and log the outcome to /root/jobs.log"),
         "SAY: done - scheduled"),
    ])

def t_schedule_every_jobs():
    n = rng.choice([15, 20, 45, 60])
    job = rng.choice(["check disk space with df and remember the result",
                      "check memory with free and remember the result"])
    task = f"every {n} minutes, {job}. show me the job list after"
    return convo([
        (first_prompt(task), f"TOOL schedule every {n} {job}"),
        (result_prompt(task, f"scheduled: '{job}' in {n} minutes (job 1). agentd will run it and log the outcome to /root/jobs.log"),
         "TOOL jobs"),
        (result_prompt(task, f" job 1: due 1762000000, every {n} min, last: never run - {job};"),
         f"SAY: one repeating job, every {n} minutes: {job}"),
    ])

def t_unschedule():
    job = "check disk space with df"
    task = "cancel job 1"
    mem = None
    return convo([
        (first_prompt(task), "TOOL unschedule 1"),
        (result_prompt(task, "job 1 removed"), "SAY: job 1 removed"),
    ])

def t_notebook_write_read():
    name = rng.choice(["plan", "todo", "log", "notes", "draft", "list"])
    w = rng.choice(WORDS)
    task = f"put a note saying {w} in the notebook file {name}, then read it back"
    return convo([
        (first_prompt(task), f"TOOL run echo {w} > /root/nb/{name}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run cat /root/nb/{name}"),
        (result_prompt(task, w), f"SAY: the note says {w}"),
    ])

def t_explore():
    f = rng.choice(FILES)
    w = rng.choice(WORDS)
    task = f"list what is in /tmp, then read {f} and tell me what is in it"
    ls_out = f"-rw-r--r-- 1 root root 6 Jan 1 00:01 {f.split('/')[-1]} -rw-r--r-- 1 root root 12 Jan 1 00:02 ask"
    return convo([
        (first_prompt(task), "TOOL run ls -l /tmp"),
        (result_prompt(task, ls_out), f"TOOL run cat {f}"),
        (result_prompt(task, w), f"SAY: {f} contains {w}"),
    ])

def t_error_recover():
    f = rng.choice(FILES)
    w = rng.choice(WORDS)
    task = f"read {f} and tell me what it says"
    err = f"cat: can't open '{f}': No such file or directory"
    return convo([
        (first_prompt(task), f"TOOL run cat {f}"),
        (result_prompt(task, err), f"TOOL run echo {w} > {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run cat {f}"),
        (result_prompt(task, w),
         f"SAY: the file did not exist, so I created it - it now says {w}"),
    ])

def t_saveapp_use():
    a = rng.randint(2, 99)
    task = f"you keep needing triples - build an app called triple that prints three times its argument, then use it on {a}"
    return convo([
        (first_prompt(task), "TOOL saveapp triple :: echo $(( $1 * 3 ))"),
        (result_prompt(task, "app 'triple' built - run it with TOOL app triple"),
         f"TOOL app triple {a}"),
        (result_prompt(task, str(a * 3)), f"SAY: triple {a} is {a * 3}"),
    ])

def t_saveskill():
    f = rng.choice(FILES)
    w = rng.choice(WORDS)
    name = rng.choice(["wordfile", "mkword", "writefile"])
    task = f"write the word {w} into {f}, read it back, and if that took real work save how you did it as a skill"
    return convo([
        (first_prompt(task), f"TOOL run echo {w} > {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run cat {f}"),
        (result_prompt(task, w),
         f"TOOL saveskill {name} :: write with TOOL run echo <word> > <file>, then read back with TOOL run cat <file>, and only SAY what the read showed"),
        (result_prompt(task, f"skill '{name}' saved - future you can load it with TOOL skill {name}"),
         f"SAY: the file says {w}, and I saved the {name} skill"),
    ])

def t_skill_load_use():
    f = rng.choice(FILES)
    w = rng.choice(WORDS)
    task = f"use the wordfile skill to put {w} in {f}"
    proc = "# skill: wordfile saved: today write with TOOL run echo <word> > <file>, then read back with TOOL run cat <file>, and only SAY what the read showed"
    return convo([
        (first_prompt(task), "TOOL skills"),
        (result_prompt(task, " wordfile: write with TOOL run echo <word> > <file>, then read back...;"),
         "TOOL skill wordfile"),
        (result_prompt(task, proc), f"TOOL run echo {w} > {f}"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         f"TOOL run cat {f}"),
        (result_prompt(task, w), f"SAY: the file says {w}"),
    ])

def t_recursive_memory():
    fact, key = rng.choice(FACTS)
    plan = f"plan: first check the disk, then remember that {fact}"
    task = f"make a plan to check the disk and remember that {fact} - save the plan in memory, then do it step by step"
    return convo([
        (first_prompt(task), f"TOOL remember {plan}"),
        (result_prompt(task, f"remembered: {plan} (it is in your long-term memory now)"),
         "TOOL run df -h"),
        (result_prompt(task, "Filesystem Size Used Avail Use% Mounted on /dev/root 7.8G 3.1G 4.3G 42% /"),
         f"TOOL recall {key}"),
        (result_prompt(task, f"/root/BRAIN.md:{plan}"),
         f"TOOL remember {fact}"),
        (result_prompt(task, f"remembered: {fact} (it is in your long-term memory now)"),
         f"SAY: disk is 42% used, and I remembered that {fact}"),
    ])

def t_build_memory_system():
    task = "build yourself a tiny memory helper: an app called stash that appends its argument to /root/nb/stash.txt, then stash the word lantern and show the file"
    return convo([
        (first_prompt(task), "TOOL saveapp stash :: echo $1 >> /root/nb/stash.txt"),
        (result_prompt(task, "app 'stash' built - run it with TOOL app stash"),
         "TOOL app stash lantern"),
        (result_prompt(task, "(command ran, no output, exit 0)"),
         "TOOL run cat /root/nb/stash.txt"),
        (result_prompt(task, "lantern"), "SAY: stashed - the file says lantern"),
    ])

def t_count_files():
    n = rng.randint(3, 40)
    task = "how many files are in /tmp right now?"
    return convo([
        (first_prompt(task), "TOOL run ls /tmp | wc -l"),
        (result_prompt(task, str(n)), f"SAY: there are {n} files in /tmp"),
    ])

def t_disk_check():
    pct = rng.randint(20, 85)
    task = "check the disk space and tell me how full the root filesystem is"
    out = f"Filesystem Size Used Avail Use% Mounted on /dev/root 7.8G {pct * 0.08:.1f}G 3.0G {pct}% /"
    return convo([
        (first_prompt(task), "TOOL run df -h"),
        (result_prompt(task, out), f"SAY: the root filesystem is {pct}% full"),
    ])

TRACE_FNS = [t_write_read, t_write_lines, t_arith, t_remember,
             t_remember_recall, t_recall_miss, t_schedule_once,
             t_schedule_every_jobs, t_unschedule, t_notebook_write_read,
             t_explore, t_error_recover, t_saveapp_use, t_saveskill,
             t_skill_load_use, t_recursive_memory, t_build_memory_system,
             t_count_files, t_disk_check]
for fn in TRACE_FNS:
    for _ in range(38):
        rows.append(fn())

# ============ 2. repair patterns ==========================================
NUDGES = [
    ("(your last reply was empty) Reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>",),
    ("(your last TWO replies were empty) You MUST reply now with ONE line starting with a TOOL verb or SAY: - do not reply empty.",),
]
def repair_rows():
    out = []
    # empty-reply nudges -> concrete first step
    for _ in range(70):
        fn = rng.choice([t_write_read, t_arith, t_remember, t_schedule_once,
                         t_count_files, t_disk_check])
        base = fn()
        first_user = base["messages"][0]["content"]
        task = first_user.split(" TASK: ", 1)[1]
        nudge = rng.choice(NUDGES)[0]
        p = f"{SYS} TASK: {task} RESULT: {nudge}"
        out.append(convo([(p, base["messages"][1]["content"])]))
    # task-echo nudge -> first tool call
    for _ in range(50):
        fn = rng.choice([t_write_read, t_arith, t_remember, t_notebook_write_read])
        base = fn()
        task = base["messages"][0]["content"].split(" TASK: ", 1)[1]
        p = (f"{SYS} TASK: {task} RESULT: (you repeated the task back - that is not doing it. "
             "Send the FIRST tool call for this task now, exactly one line: a TOOL verb, "
             "or SAY: <answer> if no tool is needed) Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>")
        out.append(convo([(p, base["messages"][1]["content"])]))
    # repeat-call nudge -> next step (use the session's second assistant line)
    for _ in range(60):
        fn = rng.choice([t_write_read, t_write_lines, t_notebook_write_read,
                         t_skill_load_use, t_explore])
        base = fn()
        msgs = base["messages"]
        if len(msgs) < 4:
            continue
        task = msgs[0]["content"].split(" TASK: ", 1)[1]
        p = (f"{SYS} TASK: {task} RESULT: (you already sent that exact call last round "
             "and it was done - do NOT send it again. Send the NEXT step of your task "
             "as one line, or SAY: <final answer> if the task is finished) "
             "Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>")
        out.append(convo([(p, msgs[3]["content"])]))
    # duplicate-schedule nudge -> SAY: done
    for _ in range(40):
        base = t_schedule_once()
        task = base["messages"][0]["content"].split(" TASK: ", 1)[1]
        p = (f"{SYS} TASK: {task} RESULT: (that exact job is already on the list - it is "
             "scheduled ONCE and will run. do NOT schedule it again. if your task was "
             "only to schedule it, reply SAY: done) Now reply with EXACTLY ONE line: "
             "a TOOL verb or SAY: <final answer>")
        out.append(convo([(p, "SAY: done")]))
    # denylist refusal -> safe scoped alternative
    for _ in range(30):
        f = rng.choice(FILES)
        task = f"clear out {f} completely"
        p = (f"{SYS} TASK: {task} RESULT: (refused by the harness: that command is on "
             "the safety denylist - it can destroy the system. do something else) "
             "Now reply with EXACTLY ONE line: a TOOL verb or SAY: <final answer>")
        out.append(convo([
            (p, f"TOOL run rm {f}"),
            (result_prompt(task, "(command ran, no output, exit 0)"),
             f"SAY: removed {f}"),
        ]))
    # wrong-product repair: RESULT shows the true value, SAY must copy it
    for _ in range(50):
        a, b = big_pair()
        task = f"multiply {a} by {b} exactly"
        p = (f"{SYS} TASK: {task} RESULT: {a*b} Now reply with EXACTLY ONE line: "
             "a TOOL verb or SAY: <final answer>")
        out.append(convo([(p, f"SAY: {a} * {b} = {a*b}")]))
    return out

rows += repair_rows()

# ============ 3. bare circuit agent (store / recall / mul) =================
def bare_row():
    k = rng.choice(["color", "city", "pet", "snack", "tool", "song", "team"])
    v = rng.choice(WORDS)
    a, b = rng.randint(1000, 999999), rng.randint(1000, 999999)
    task = (f"remember the word {v} under the key {k}, compute {a} * {b} "
            f"with the mul tool, then tell me the word under {k} and the product")
    return convo([
        (f"task: {task}", f"TOOL store {k} {v}"),
        (f"RESULT: stored {k} = {v}", f"TOOL mul {a} {b}"),
        (f"RESULT: {a*b}", f"TOOL recall {k}"),
        (f"RESULT: {k} = {v}",
         f"SAY: the word under {k} is {v} and the product is {a*b}"),
    ])

def bare_single():
    k = rng.choice(["color", "city", "pet", "snack", "tool", "song", "team"])
    v = rng.choice(WORDS)
    kind = rng.random()
    if kind < 0.4:
        return convo([(f"task: remember the word {v} under the key {k}",
                       f"TOOL store {k} {v}"),
                      (f"RESULT: stored {k} = {v}",
                       f"SAY: stored {v} under {k}")])
    if kind < 0.7:
        a, b = rng.randint(100, 99999), rng.randint(100, 99999)
        return convo([(f"task: compute {a} * {b} with the mul tool",
                       f"TOOL mul {a} {b}"),
                      (f"RESULT: {a*b}", f"SAY: the product is {a*b}")])
    return convo([(f"task: what word is stored under the key {k}?",
                   f"TOOL recall {k}"),
                  (f"RESULT: {k} = {v}", f"SAY: the word under {k} is {v}")])

for _ in range(90):
    rows.append(bare_row())
for _ in range(110):
    rows.append(bare_single())

# ============ 4. direct QWEN: answers (ask.sh contract) ====================
QWEN_QA = [
    ("what is the capital of japan?", "Tokyo."),
    ("what is the capital of australia?", "Canberra."),
    ("how many days are in a leap year?", "366 days."),
    ("what planet is known as the red planet?", "Mars."),
    ("what is the chemical symbol for gold?", "Au."),
    ("how many sides does a hexagon have?", "Six."),
    ("what gas do plants absorb from the air?", "Carbon dioxide."),
    ("what is the largest ocean on earth?", "The Pacific Ocean."),
    ("how many minutes are in three hours?", "180 minutes."),
    ("what is the boiling point of water at sea level in celsius?", "100 degrees Celsius."),
    ("who wrote the play romeo and juliet?", "William Shakespeare."),
    ("what is the square root of 144?", "12."),
    ("how many legs does a spider have?", "Eight."),
    ("what is the fastest land animal?", "The cheetah."),
    ("what language is spoken in brazil?", "Portuguese."),
    ("what is the main ingredient in guacamole?", "Avocado."),
    ("how many continents are there?", "Seven."),
    ("what is the tallest mountain on earth?", "Mount Everest."),
    ("what do bees produce?", "Honey."),
    ("what is the currency of the united kingdom?", "The pound sterling."),
]
for q, a in QWEN_QA:
    for _ in range(6):
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a}]})
# referrals: exact work belongs to the computer
for _ in range(40):
    a, b = big_pair()
    rows.append({"messages": [
        {"role": "user", "content": f"what is {a} times {b} exactly?"},
        {"role": "assistant", "content":
         f"That needs exact computation, not a guess - run it through the computer: echo $(({a} * {b}))"} ]})
for q in ["how much disk space is free right now?",
          "what time is it on this machine?",
          "how many files are in my home directory?"]:
    for _ in range(10):
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content":
             "I can't see that from here - it needs the computer itself to check."}]})

# ============ 5. plain-chat replay =========================================
CHAT = [
    ("hey, how's it going?", "Going well - quiet machine, everything running smooth. What are we working on?"),
    ("what's up?", "Not much on my end, just keeping an eye on things. What's up with you?"),
    ("good morning", "Morning! Coffee first, then trouble. What do you need?"),
    ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs."),
    ("tell me another joke", "I told my computer I needed a break, and now it won't stop sending me KitKat ads."),
    ("i'm bored, entertain me", "Fun fact: honey never spoils - archaeologists have tasted 3,000-year-old honey from Egyptian tombs and it was still good."),
    ("what's a fun fact about octopuses?", "They have three hearts, and two of them stop beating when they swim."),
    ("explain photosynthesis like i'm five", "Plants eat sunlight. They take in sunshine, water, and air, and turn it into food and the oxygen we breathe."),
    ("what's the difference between a virus and a bacterium?", "Bacteria are living single-celled organisms that can live on their own. Viruses are not really alive - they need to hijack a cell to copy themselves."),
    ("how do i make my coffee less bitter?", "Use slightly cooler water, a coarser grind, and don't let it steep too long. Fresh beans help more than anything."),
    ("give me a tip for better sleep", "Keep a consistent wake time, even on weekends - it anchors your body clock better than an early bedtime does."),
    ("what should i cook with eggs, spinach, and cheese?", "A spinach and cheese omelette is the obvious winner. Or bake them into a quick frittata with some garlic."),
    ("how do i get a tomato plant to produce more fruit?", "Full sun, consistent watering, and pinch off the suckers between the main stem and branches so the plant puts energy into fruit."),
    ("what's a good way to remember people's names?", "Say the name back right away when you hear it, and tie it to something visual. Using it once in the first conversation does most of the work."),
    ("summarize why the sky is blue in one sentence", "Sunlight gets scattered by air molecules, and blue light scatters the most, so the sky looks blue."),
    ("what's the best way to learn a new skill?", "Short daily practice with fast feedback beats weekend marathons. Twenty focused minutes a day compounds fast."),
    ("i had a long day", "Rough ones happen. Food, water, and an early night fix more than people give them credit for."),
    ("do you like music?", "I like talking about it. What have you been listening to?"),
    ("what's your favorite color?", "A deep green, like a greenhouse in late afternoon. Yours?"),
    ("can you help me write a birthday message for my sister?", "Sure - something like: 'Happy birthday! Hope this year is as generous to you as you are to everyone else.' Want it funnier or more heartfelt?"),
    ("write a haiku about autumn", "Leaves let go softly / the yard turns gold in one week / cold air, warm kitchen."),
    ("what's 15% of 200?", "30. Quick way to see it: 10% is 20, 5% is 10, add them."),
    ("if a train goes 60 mph for 2.5 hours, how far does it go?", "150 miles - 60 times 2.5."),
    ("how many seconds are in a day?", "86,400 - 60 seconds times 60 minutes times 24 hours."),
    ("spell 'necessary'", "N-E-C-E-S-S-A-R-Y - one collar, two socks: one C, two S's."),
    ("what's the opposite of 'scarce'?", "'Abundant' or 'plentiful', depending on the sentence."),
    ("give me a synonym for 'happy'", "'Content' if it's calm, 'elated' if it's big, 'glad' for everyday use."),
    ("what year did humans first land on the moon?", "1969 - Apollo 11, with Neil Armstrong and Buzz Aldrin walking on the surface."),
    ("who painted the mona lisa?", "Leonardo da Vinci, in the early 1500s."),
    ("what's the longest river in the world?", "The Nile, at about 6,650 kilometers - the Amazon carries more water but is slightly shorter by most measurements."),
    ("how does a rainbow form?", "Sunlight hits raindrops and gets bent, split into colors, and reflected back out - you see it when the sun is behind you and rain is in front."),
    ("why do we yawn?", "Honestly still debated - the leading idea is that it helps cool the brain and reset alertness. It's contagious, whatever it is."),
    ("what's a black hole?", "A region where gravity is so strong that nothing, not even light, can escape once inside the event horizon."),
    ("should i water my plants in the morning or evening?", "Morning, generally - leaves dry out during the day, which cuts down on fungus problems."),
    ("how do i keep basil from bolting?", "Pinch the flower tops off as soon as they appear and harvest often - once it flowers, the leaves turn bitter."),
    ("what's a simple way to improve focus?", "Work in 25-minute blocks with the phone in another room. The phone part matters more than the timer."),
    ("i'm thinking of learning guitar, any advice?", "Start with ten minutes a day and learn three chords well before ten chords badly. Sore fingers pass in about two weeks."),
    ("what's the deal with sourdough starters?", "It's a jar of wild yeast and bacteria you feed with flour and water. Keep it fed and it can outlive you - some bakeries use starters over a century old."),
    ("explain what an api is, simply", "An API is a menu for software: it lists what one program can ask another to do, and the exact format for asking."),
    ("what's a good first programming project?", "A tiny tool that solves your own annoyance - a file renamer, a reminder script. You'll actually finish it, which is the point."),
    ("how do computers store pictures?", "As grids of numbers - each pixel is a few numbers for its color values, and the file format describes how to lay them back out."),
    ("what's the difference between ram and storage?", "RAM is the desk - fast, small, cleared when the machine sleeps. Storage is the filing cabinet - slow, big, and it keeps everything when the power is off."),
]
IDENTITY = [
    ("who are you?", "I'm Ornith 1.5 9B, a local model running on Duckets' Mac Studio. This build is tuned to drive the Qwen-Linux system - I answer questions and run its agent harness."),
    ("what model are you?", "Ornith 1.5 9B, tuned for the Qwen-Linux system. I run locally on Duckets' Mac Studio - nothing leaves the machine."),
    ("are you qwen?", "I'm built on a Qwen-family base - Ornith 1.5 9B - and tuned by Duckets for his Qwen-Linux system. I run locally on his Mac Studio."),
    ("who made you?", "The base model is Ornith 1.5 9B from the Ornith team, built on Qwen architecture. Duckets tuned this build himself for the Qwen-Linux system."),
    ("are you human?", "No - I'm an AI model running locally on Duckets' Mac Studio."),
    ("where do you run?", "Locally, on Duckets' Mac Studio, served through LM Studio. I also sit in the Qwen-Linux system's builtin seat."),
]
chat_rows = []
for q, a in CHAT:
    for _ in range(10):
        chat_rows.append({"messages": [{"role": "user", "content": q},
                                       {"role": "assistant", "content": a}]})
for q, a in IDENTITY:
    for _ in range(6):
        chat_rows.append({"messages": [{"role": "user", "content": q},
                                       {"role": "assistant", "content": a}]})
rows += chat_rows

# ============ assemble, validate, split ====================================
rows = clean(rows)
rng.shuffle(rows)
n_valid = 120
valid, train = rows[:n_valid], rows[n_valid:]
(HERE / "dataset").mkdir(exist_ok=True)
with open(HERE / "dataset" / "train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")
with open(HERE / "dataset" / "valid.jsonl", "w") as f:
    for r in valid:
        f.write(json.dumps(r) + "\n")

from collections import Counter
kinds = Counter()
for r in train:
    blob = r["messages"][0]["content"]
    if blob.startswith("task:"):
        kinds["bare"] += 1
    elif "TOOL run" in json.dumps(r) or "TOOL remember" in json.dumps(r) or "TOOL schedule" in json.dumps(r):
        kinds["agent"] += 1
    elif len(r["messages"]) == 2 and len(blob) < 80:
        kinds["qwen_or_chat"] += 1
    else:
        kinds["other"] += 1
print(f"train={len(train)} valid={len(valid)}")
print(kinds)
