"""Build the portable HTML from the task ledger and locally verified run artifacts."""
import datetime as dt
from html import escape
import json
import os
from pathlib import Path
import tempfile
try:
    from .measurements import METHODS, TABLE, read_cases, report_figure
except ImportError:
    from measurements import METHODS, TABLE, read_cases, report_figure
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tracker"
RESULTS = ROOT / "results"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def set_state(task, state, evidence):
    task[2:] = [state, evidence]


def render():
    phases = read_json(HERE / "tasks.json")
    tasks = {t[0]: t for group in phases for t in group["tasks"]}
    environment_files = [RESULTS / "environment.json", RESULTS / "host-probe.json", RESULTS / "workspace-environment.json"]
    for data in (read_json(f) for f in environment_files):
        p = data.get("gpu_probe", {})
        if p.get("passed") is True and p.get("verified_outputs") == 128:
            set_state(tasks["Confirm GPU execution on this Mac"], "Complete",
                      f"Verified {p['device']} with 128 correct outputs, {p['threadgroup_bytes_used']:,} bytes of threadgroup memory and runtime Metal compilation.")
            break
    cells, sessions = read_cases(RESULTS)
    figure, combined = report_figure(RESULTS, sessions)
    cells.update(combined)
    verified_cases = {case for _,_,_,items in sessions for case in items}
    for title, suite, expected_count in (
        ("Run the two-point pilot","pilot",2),
        ("Run the full attention sweep","attention",7),
        ("Time the connected two-layer stack","stack",2)):
        relevant = [run for run in sessions if run[1] in (suite,"full")]
        if relevant:
            session,suite_name,status,items = relevant[-1]
            if suite=="stack":
                known = [x for x in items if x.startswith("stack-")]
            elif suite=="pilot":
                known = [x for x in items if x in ("attention-b1-l64","attention-b1-l4096")]
            else:
                known = [x for x in items if x.startswith("attention-")]
            set_state(tasks[title], "In progress" if len(known)<expected_count or suite=="pilot" else "Complete",
                      f"{len(known)} of {expected_count} cases have paired timings and a parity check in {session}. Run status: {status}. Review traces separately.")
    both_pilot_cases = {"attention-b1-l64","attention-b1-l4096"} <= verified_cases
    if both_pilot_cases:
        for name in ("Build staged Metal","Build the single-dispatch Metal kernel","Build the hybrid schedule"):
            set_state(tasks[name],"Complete","Output, updated K/V and histories passed the CPU parity gate before and after device-resident timings at 64 and 4,096 visible positions.")
        set_state(tasks["Build the compiled MLX baseline"],"In progress","The compiled path passed checked GPU trials at both pilot lengths. Confirm the selected SDPA dispatches in a trace.")
        set_state(tasks["Check the parallelism explanation"],"In progress","The split-context schedule ran at the two pilot lengths. Inspect the marked GPU intervals before attributing differences to parallelism.")
    if combined:
        set_state(tasks["Repeat and audit the measurements"],"In progress","Two compatible sessions have been measured. Review thermal conditions, per-session winners, confidence intervals and trace evidence before interpreting the results.")
        set_state(tasks["Render the figure and results table"],"Complete",
                  "The independent-session report generated the plotted latency intervals and table from checked raw trials.")
    table_rows=[]
    for scope,batch,length in TABLE:
        case=f"{scope}-b{batch}-l{length}"
        label="Global attention" if scope=="attention" else "Two-layer stack"
        vals=[f'<td>{cells[case,m]:,.2f}</td>' if (case,m) in cells else '<td class="na">—</td>' for m in METHODS]
        table_rows.append(f'<tr><td>{label}</td><td>{batch}</td><td>{length:,}</td>'+''.join(vals)+'</tr>')
    headnote=""
    if combined:
        headnote='<p class="note">Combined medians from the verified independent-session report.</p>'
    elif verified_cases:
        headnote='<p class="note">Preliminary medians from the latest parity-checked session for each case. The final figure requires an independent repeat.</p>'
    cpu=read_json(RESULTS/"cpu-attention.json")
    stack=read_json(RESULTS/"cpu-decoder.json")
    checkpoints=[]
    if cpu.get("device")=="CPU":
        error=cpu["comparisons"]["gated_output"]["max_abs"]
        checkpoints.append(f"At 48 Q heads, 8 KV heads and 128 channels per head, the 64-token CPU trajectory matched the independent prefix calculation within {error:.8f} maximum absolute output error.")
    if stack.get("method")=="cpu" and stack.get("geometry",{}).get("width")==1152 and stack.get("output_arrays")==15:
        checkpoints.append(f"The connected decoder ran at width 1,152. Both layers routed to eight of 384 experts, with {stack['output_arrays']} finite output and state arrays.")
    proofbox=('<div class="callout"><b>Verified on CPU</b><p>'+escape(' '.join(checkpoints))+'</p></div>') if checkpoints else ''
    escaped_groups = []
    for group in phases:
        details = []
        for t in group["tasks"]:
            name, criterion = map(escape, t[:2])
            st = t[2] if len(t) > 2 else "Planned"
            evidence = '<p class="evidence">'+escape(t[3])+'</p>' if len(t)>3 else ''
            klass = {"Complete":"done","In progress":"running"}.get(st,"planned")
            mark = "✓" if st == "Complete" else ""
            details.append(f'<details class="task {klass}"><summary><span class="mark" aria-hidden="true">{mark}</span>'
                           f'<span>{name}</span><span class="phase">{escape(st)}</span></summary><p>{criterion}</p>{evidence}</details>')
        escaped_groups.append('<div class="group"><h3>'+escape(group["title"])+'</h3>'+''.join(details)+'</div>')
    items = [t for group in phases[1:] for t in group["tasks"]]
    done = sum(len(t)>2 and t[2]=="Complete" for t in items)
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
    timestamp = now.strftime("Updated %b %d, %Y · %H:%M %Z")
    repl = {"@@TASKS@@": json.dumps(phases, ensure_ascii=False).replace("<",r"\u003c"),
            "@@STATIC@@": ''.join(escaped_groups), "@@SUMMARY@@": "GPU report available" if combined else ("GPU measurements underway" if sessions else "CPU verified · GPU pending"),
            "@@COUNT@@": f"{done} of {len(items)} execution tasks complete", "@@STAMP@@":timestamp,
            "@@RESULT_LINKS@@":headnote, "@@RESULT_ROWS@@":"".join(table_rows),
            "@@PERF_FIGURE@@": figure, "@@CHECKPOINTS@@":proofbox}
    page = (HERE / "template.html").read_text()
    for marker, value in repl.items():
        page = page.replace(marker,value)
    path=ROOT/"dist/index.html"
    path.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        tmp.write(page)
    os.chmod(tmp.name,0o644)
    os.replace(tmp.name,path)
    return path


if __name__=="__main__":
    print(render())
