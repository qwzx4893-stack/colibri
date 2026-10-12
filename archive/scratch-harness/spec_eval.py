import statistics, sys
sys.path.insert(0, r"C:\Users\STS\lma_ab")
from gpu_run import serve_run
N = int(sys.argv[1]); reps = int(sys.argv[2])
cfgs = [("plain", {"LMA_GPU": "1", "LMA_SPEC": "0"})]
for a in sys.argv[3:]:
    name, _, kv = a.partition(":"); cfgs.append((name, dict(dict(x.split("=", 1) for x in kv.split(",") if x), LMA_GPU="1")))
DOC = ("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
       "Sphinx of black quartz, judge my vow. How vexingly quick daft zebras jump. ")
CODE = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n"
W = {
 "chat": "Explain how a hash table works, covering hashing, collisions and resizing.",
 "code": "Here is a Python file:\n" + CODE + "\nAdd type hints and a docstring to every function and return the full file.",
 "copy": "Repeat the following text exactly three times, separated by blank lines:\n" + DOC,
 "json": 'Convert this list to a JSON array of objects with fields "name" and "age": Alice 30, Bob 25, Carol 41, Dave 19, Eve 35.',
 "summ": "Summarise in two sentences: " + DOC * 3,
}
res = {}
for w, prompt in W.items():
    rows = {n: [] for n, _ in cfgs}; texts = {n: set() for n, _ in cfgs}; info = {}
    for r in range(reps):
        for n, env in cfgs:
            x = serve_run(env, N, prompt=prompt, warm=True)
            rows[n].append(x["tok_s"]); texts[n].add(x["text"]); info[n] = x.get("spec_line")
    base = texts["plain"]
    print(f"== {w}")
    for n, _ in cfgs:
        print(f"  {n:10s} median {statistics.median(rows[n]):6.2f} tok/s (min {min(rows[n]):.1f} max {max(rows[n]):.1f}) identical_to_plain={texts[n]==base}")
        if info[n] and n != "plain": print("    ", info[n])
    sys.stdout.flush()
