"""Build the portable HTML from the task ledger and locally verified run artifacts."""
import datetime as dt
from html import escape
import json
import os
import re
import sys
from pathlib import Path
import tempfile
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker.measurements import diagnostic_reports, final_report, preliminary_figure, read_runs
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


def stack_failure(run):
    if not run or run.status != "incomplete":
        return ""
    field = re.search(r"\b(layer(?:10|11)_(?:\w+)|summary)\b", run.error)
    if not field:
        return ""
    message = f"The two-layer comparison in {run.name} stopped at {field[0]}."
    count = re.search(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)", run.error)
    error = re.search(r"Max absolute difference among violations:\s*([\d.eE+-]+)", run.error)
    if count and error:
        message += (f" {int(count[1]):,} of {int(count[2]):,} values exceeded the registered tolerance. "
                    f"The largest absolute difference among them was {error[1]}.")
    return message


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
    runs = read_runs(RESULTS)
    cohort = [r for r in runs if r.compatible(runs[-1])] if runs else []
    # Display one sweep even when an incomplete repeat has fewer verified cases.
    selected = max((r for r in cohort if r.cases),
                   key=lambda r: (len(r.cases), r.started), default=None)
    pilot = next((r for r in reversed(runs) if r.suite == "pilot" and r.status == "complete"), None)
    sweep = next((r for r in reversed(cohort) if r.suite in ("attention", "full")), None)
    stack_run = next((r for r in reversed(cohort) if r.suite in ("stack", "full")), None)
    failure = stack_failure(stack_run)
    report = final_report(RESULTS, cohort)
    attention_report = None if report else final_report(RESULTS, cohort, "attention")
    diagnostics = diagnostic_reports(RESULTS, cohort[-1] if cohort else None)
    replay_failed = any(statuses for _, _, statuses in diagnostics)
    history = []
    cells, figure, headnote = {}, "", ""
    summary = "CPU verified · GPU pending"
    if pilot and len(pilot.cases) == 2:
        set_state(tasks["Run the two-point pilot"], "Complete",
                  f"{pilot.name} verified timings at both 64 and 4,096 tokens. Trace review is tracked separately.")
        if pilot not in cohort:
            history.append(f"The earlier pilot {pilot.name} verified both endpoint lengths.")
    previous_failure = next((r for r in reversed(runs) if r not in cohort and stack_failure(r)), None)
    if previous_failure:
        history.append("Earlier run. " + stack_failure(previous_failure))
    for run, prefix, title, total in (
            (sweep, "attention-", "Run the full attention sweep", 7),
            (stack_run, "stack-", "Time the connected two-layer stack", 2)):
        if run:
            count = sum(case.startswith(prefix) for case in run.cases)
            state = "Complete" if count == total else "In progress"
            evidence = f"{run.name} has verified timings for {count} of {total} cases."
            if prefix == "stack-" and failure:
                state = "Blocked"
                evidence += " " + failure
            set_state(tasks[title], state, evidence)
    if diagnostics:
        lines = []
        for length, seed, failures in diagnostics:
            outcome = " · ".join(failures) if failures else "all four schedules passed"
            lines.append(f"At {length:,} tokens and seed {seed}, {outcome}.")
        diagnostic_html = ('<div class="callout"><b>Two-layer GPU replay</b><p>'
                           + '</p><p>'.join(escape(line) for line in lines)
                           + '</p></div>')
        if replay_failed:
            set_state(tasks["Time the connected two-layer stack"], "Blocked",
                      "A replay exceeded the registered numerical tolerance. See the case, seed and schedules above.")
    else:
        diagnostic_html = ""
    if selected:
        ends = {"attention-b1-l64", "attention-b1-l4096"}
        if any(ends <= r.cases.keys() for r in cohort):
            for name in ("Build staged Metal", "Build the single-dispatch Metal kernel", "Build the hybrid schedule"):
                set_state(tasks[name], "Complete", "Output, updated K/V and convolution histories passed the CPU comparison before and after timed runs at both endpoint lengths.")
            set_state(tasks["Build the compiled MLX baseline"], "In progress", "Compiled MLX passed both endpoint checks. Its SDPA dispatches await trace review.")
            set_state(tasks["Check the parallelism explanation"], "In progress", "The split-context control ran at both endpoint lengths. The timing data are available. GPU occupancy and MLX dispatch counts await trace review.")
        cells = {(name, method): value for name, methods in selected.cases.items() for method, value in methods.items()}
        figure = preliminary_figure(ROOT / "analysis/attention-preliminary.svg", selected, cohort)
        headnote = (f"Preliminary medians from {selected.name} on {selected.spec['machine']['device']}. "
                    f"Each reported schedule has {selected.spec['trials']} paired timing blocks across "
                    f"{selected.spec['fixtures']} seeded states.")
        if failure:
            headnote += " " + failure
        summary = "Attention measured · stack check failed" if failure or replay_failed else "GPU measurements underway"
        if figure:
            set_state(tasks["Render the figure and results table"], "In progress",
                      "The preliminary attention table and figure use verified trials. The combined report requires two complete sessions.")
        if not report:
            complete = {}
            for r in cohort:
                if r.status == "complete":
                    complete.setdefault(r.name, set()).update(r.cases)
            count = sum(len(cases) == 9 for cases in complete.values())
            set_state(tasks["Repeat and audit the measurements"], "In progress",
                      f"{count} of 2 full sessions have completed with this benchmark configuration. Instruments trace review is pending.")
    if attention_report:
        cells, figure, names, traces = attention_report
        summary = "Attention repeated · stack check failed" if replay_failed or failure else "Attention repeated · stack pending"
        headnote = (f"Attention medians combine {len(names)} verified sessions, {', '.join(names)}. "
                    "Each stack cell will appear in the combined report once both sessions pass its correctness checks.")
        if failure:
            headnote += " " + failure
        set_state(tasks["Run the full attention sweep"], "Complete",
                  f"All seven attention cases have independent repeats in {', '.join(names)}.")
        set_state(tasks["Render the figure and results table"], "In progress",
                  "The attention figure and table combine two verified runs. Two-layer latency is pending.")
    if report:
        cells, figure, names, traces = report
        summary = "GPU report available"
        headnote = f"Combined medians from {', '.join(names)}. The report includes {len(traces)} inspected dispatch counts."
        set_state(tasks["Repeat and audit the measurements"], "In progress",
                  "The compatible sessions are complete. Review thermal conditions, per-session rankings and trace coverage.")
        set_state(tasks["Render the figure and results table"], "Complete",
                  "The independent-session report generated the plotted latency intervals and table from verified trials.")
    table_rows = []
    for scope, batch, length in (("attention",1,64), ("attention",1,4096), ("attention",8,4096),
                                  ("stack",1,64), ("stack",1,4096)):
        case = f"{scope}-b{batch}-l{length}"
        label = "Global attention" if scope == "attention" else "Two-layer stack"
        vals = [f'<td>{cells[case,m]:,.1f}</td>' if (case,m) in cells else '<td class="na">—</td>'
                for m in ("compiled", "staged", "hybrid", "single", "split")]
        table_rows.append(f'<tr><td>{label}</td><td>{batch}</td><td>{length:,}</td>'+''.join(vals)+'</tr>')
    if headnote:
        headnote = '<p class="note">' + escape(headnote) + '</p>'
    if not figure:
        figure = ('<p class="note">The figure will show batch-one latency across all six context lengths. '
                  'The combined report requires an independent repeat.</p>')
    cpu=read_json(RESULTS/"cpu-attention.json")
    stack=read_json(RESULTS/"cpu-decoder.json")
    checkpoints=[]
    if cpu.get("device")=="CPU":
        error=cpu["comparisons"]["gated_output"]["max_abs"]
        checkpoints.append(f"At 48 Q heads, 8 KV heads and 128 channels per head, the 64-token CPU trajectory matched the independent prefix calculation within {error:.8f} maximum absolute output error.")
    if stack.get("method")=="cpu" and stack.get("geometry",{}).get("width")==1152 and stack.get("output_arrays")==15:
        checkpoints.append(f"The connected decoder ran at width 1,152. Both layers routed to eight of 384 experts, with {stack['output_arrays']} finite output and state arrays.")
    proofbox=('<div class="callout"><b>Verified on CPU</b><p>'+escape(' '.join(checkpoints))+'</p></div>') if checkpoints else ''
    history_html = ('<p class="note">' + escape(" ".join(history)) + '</p>') if history else ""
    proofbox = diagnostic_html + history_html + proofbox
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
            "@@STATIC@@": ''.join(escaped_groups), "@@SUMMARY@@": summary,
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
