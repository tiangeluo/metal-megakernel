"""Build the portable HTML from the task ledger and locally verified run artifacts."""
import datetime as dt
from html import escape
from itertools import chain
import json
import os
import re
import sys
from pathlib import Path
import tempfile
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracker.measurements import diagnostic_reports, final_report, followup_status, inspected_trace_rows, preliminary_figure, read_runs, repeated_figure, retry_trace_status
from tracker.precision import precision_panel, precision_results
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


def replay_markup(records):
    if not records:
        return ""
    methods = ("compiled", "staged", "hybrid", "single")
    rows = []
    paired = []
    route_and_summary_passed = True
    matched_checks = []
    for length, seed, failures, comparisons, stage in records:
        cells = "".join(f"<td>{escape(failures.get(m, 'Pass'))}</td>" for m in methods)
        rows.append(f"<tr><td>{length:,}</td><td>{seed}</td>{cells}</tr>")
        for method in methods:
            obs = comparisons.get(method, {}).get("vs_cpu", {}).get("observables", {})
            for key in ("layer10_route_ids", "layer11_route_ids", "summary"):
                regions = obs.get(key, {}).get("regions", {})
                if not regions or any(r["violations"] for r in regions.values()):
                    route_and_summary_passed = False
        if failures:
            paired.append({m: comparisons.get(m, {}).get("vs_compiled", {}) for m in methods[1:]})
            layers = stage.get("findings", {}).get("layers", {})
            if stage.get("method") == "compiled" and layers:
                def outcomes(tree):
                    if isinstance(tree, dict):
                        if "passed" in tree:
                            yield tree["passed"]
                        for value in tree.values():
                            yield from outcomes(value)
                checks = list(outcomes(layers))
                if checks and all(checks):
                    matched_checks.append(len(checks))
    notes = []
    if route_and_summary_passed:
        notes.append("Every schedule matched both layers’ expert IDs exactly and passed the tolerance check for the final residual.")
    if paired and all(p["hybrid"].get("passed") and all(
            p[m].get("failed_observables") == ["layer11_post_moe"] for m in ("staged", "single")) for p in paired):
        notes.append("On the failing fixtures, hybrid also passed against compiled MLX. Staged and single each differed "
                     "from compiled MLX in the global layer’s expert-output history.")
    if paired and len(matched_checks) == len(paired) and len(set(matched_checks)) == 1:
        notes.append(f"All {matched_checks[0]} local checks on the compiled path passed for each failing fixture "
                     "when NumPy received the same stage inputs. The source of the end-to-end difference remains open.")
    return ('<h3 class="subhead">Original FP16 expert stack</h3><p class="note">Each row replays one fixed token through '
            'both layers. Pass means that every registered output agreed with the CPU reference within its tolerance. '
            'Fractions count entries outside tolerance in the global layer’s expert-output history.</p>'
            '<div class="scroll"><table><thead><tr><th>Visible tokens</th><th>Seed</th><th>Compiled MLX</th>'
            '<th>Staged</th><th>Hybrid</th><th>Single</th></tr></thead><tbody>' + "".join(rows) +
            '</tbody></table></div>' + "".join(f'<p class="note">{escape(n)}</p>' for n in notes))


def render():
    phases = read_json(HERE / "tasks.json")
    tasks = {t[0]: t for group in phases for t in group["tasks"]}
    environment_files = [RESULTS / "environment.json", RESULTS / "host-probe.json", RESULTS / "workspace-environment.json"]
    runs = read_runs(RESULTS)
    # The release carries the acquisition probe inside each session manifest.
    # This also lets the tracker rebuild on a machine without a local GPU.
    environments = (read_json(f) for f in environment_files)
    session_environments = (
        read_json(RESULTS / run.name / run.suite / "manifest.json").get("environment", {})
        for run in reversed(runs) if run.cases
    )
    for data in chain(environments, session_environments):
        p = data.get("gpu_probe", {})
        if p.get("passed") is True and p.get("verified_outputs") == 128:
            set_state(tasks["Confirm GPU execution on this Mac"], "Complete",
                      f"Verified {p['device']} with 128 correct outputs, {p['threadgroup_bytes_used']:,} bytes of threadgroup memory and runtime Metal compilation.")
            break
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
    replay_failed = any(record[2] for record in diagnostics)
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
    diagnostic_html = replay_markup(diagnostics)
    parent = next((r for r in cohort if r.name == "m4pro-02-a" and r.suite == "attention" and r.status == "complete"), None)
    precision_cells, precision_source, precision_note = precision_results(RESULTS)
    trace_path, trace_methods = retry_trace_status(RESULTS, parent)
    if diagnostics and parent:
        status, outcomes = followup_status(RESULTS, parent)
        table = ""
        if outcomes:
            rows = "".join(f"<tr><td>{seed}</td><td>{length:,}</td><td>{rc} / {rs}</td><td>{hc} / {hs}</td></tr>"
                           for seed, length, (rc, rs, hc, hs) in outcomes)
            table = ('<p>Each cell counts expert channels outside tolerance for compiled MLX / staged Metal.</p>'
                     '<div class="scroll"><table><thead><tr><th>Seed</th><th>Tokens</th><th>GPU residual only</th>'
                     f'<th>GPU heads only</th></tr></thead><tbody>{rows}</tbody></table></div>')
        diagnostic_html += ('<div class="callout"><b>Layer 11 crossover</b><p>The replay swaps the incoming residual '
                            'and attention heads independently between CPU, compiled MLX and staged Metal. The same CPU '
                            'layer finish evaluates all nine input pairs to test which input carries the drift.</p>'
                            f'{table}<p>{escape(status)}</p></div>')
    if replay_failed:
        set_state(tasks["Time the connected two-layer stack"], "Blocked",
                  "A replay exceeded the registered numerical tolerance. The table above identifies the affected seed, length and schedules.")
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
        detailed = repeated_figure(ROOT / "analysis/attention-verified.svg", cohort, names)
        figure = detailed or figure
        summary = "Attention repeated · stack check failed" if replay_failed or failure else "Attention repeated · stack pending"
        headnote = (f"Attention medians combine {len(names)} verified sessions, {', '.join(names)}. "
                    f"Each attention cell contains {len(names)*selected.spec['trials']} paired timing blocks across "
                    f"{selected.spec['fixtures']} fixed token states.")
        key = "attention-b1-l4096"
        if (key, "split") in cells and (key, "single") in cells:
            headnote += (f" At 4,096 tokens, dividing the KV scan into chunks reduced median wall time from "
                         f"{cells[key, 'single']:,.0f} to {cells[key, 'split']:,.0f} µs "
                         f"({cells[key, 'single']/cells[key, 'split']:.2f}×).")
        if failure:
            headnote += " " + failure
        set_state(tasks["Run the full attention sweep"], "Complete",
                  f"All seven attention cases have independent repeats in {', '.join(names)}.")
        set_state(tasks["Render the figure and results table"], "In progress",
                  "The attention figure and table combine two verified runs. Two-layer latency is pending.")
        set_state(tasks["Repeat and audit the measurements"], "In progress",
                  f"Both attention sessions are complete and their raw trials passed verification. Connected-stack timings and Instruments traces remain open.")
    if report:
        cells, figure, names, traces = report
        summary = "GPU report available"
        headnote = f"Combined medians from {', '.join(names)}. The report includes {len(traces)} inspected dispatch counts."
        set_state(tasks["Repeat and audit the measurements"], "In progress",
                  "The compatible sessions are complete. Review thermal conditions, per-session rankings and trace coverage.")
        set_state(tasks["Render the figure and results table"], "Complete",
                  "The independent-session report generated the plotted latency intervals and table from verified trials.")
    if precision_cells:
        set_state(tasks["Time the connected two-layer stack"], "Complete",
                  "Two independent sessions timed the connected decoder at both lengths with FP32 expert arithmetic. "
                  "The original FP16 expert stack exceeded tolerance and has no verified latency.")
        if attention_report or report:
            summary = "Attention and two-layer control measured"
            set_state(tasks["Repeat and audit the measurements"], "Complete",
                      "Attention and the FP32 expert control each have two verified sessions, paired timing blocks and confidence intervals. "
                      "The original stack’s failed fixtures are preserved.")
            set_state(tasks["Write the post and prepare the code link"], "In progress",
                      "The article draft and source include the measured workloads and the original parity limitation. "
                      "Homepage publication and the public repository link are pending.")
            combined_figure = repeated_figure(ROOT / "analysis/decoder-performance.svg", cohort, names, precision_source)
            if combined_figure:
                figure = combined_figure
                set_state(tasks["Render the figure and results table"], "Complete",
                          "The primary figure and table use verified raw observations from both workloads.")
    dispatch_note = ("Native code encodes three dispatches for staged Metal, one for single-dispatch Metal and four "
                     "for the split control. Framework dispatch counts require inspection of the marked interval.")
    if trace_methods:
        methods = ", ".join(sorted(trace_methods))
        dispatch_note += (f" Instruments recorded {len(trace_methods)} parity-checked long-context captures ({methods}). "
                          "The dispatches inside the marked decode intervals await inspection.")
        set_state(tasks["Check the parallelism explanation"], "In progress",
                  f"Captured {len(trace_methods)} schedules at batch one and 4,096 tokens. "
                  "Threadgroup geometry and GPU times are documented. Achieved occupancy requires trace evidence.")
    inspected = inspected_trace_rows(RESULTS, trace_path, trace_methods)
    if len(inspected) == 4 and all(r["command_buffers"] == r["compute_encoders"] == r["gpu_compute_intervals"] == 1
                                   and r["observed_dispatches"] is None for r in inspected.values()):
        dispatch_note = ("We inspected four separate Instruments captures at batch one and 4,096 tokens. Each marked "
                         "decode step contained one command buffer, one compute encoder and one GPU compute interval. "
                         "Operations can share an encoder. These traces leave individual dispatch counts and occupancy unresolved. "
                         "The native code submits one dispatch for single, three for staged and four for split.")
        set_state(tasks["Build the compiled MLX baseline"], "Complete",
                  "The compiled MLX attention schedule passed both endpoint checks. Its trace resolves one compute "
                  "encoder. The internal SDPA shader and dispatch count remain unverified.")
        set_state(tasks["Check the parallelism explanation"], "Complete",
                  "The split scan exposes more parallel work at long context. Reviewed four marked traces plus "
                  "native command-buffer timings. The trace granularity leaves achieved occupancy unresolved.")
        if precision_cells and (attention_report or report):
            summary = "Experiments complete · article drafted"
    table_rows = []
    display_rows = [("attention", 1, n, "Global attention", cells) for n in (64, 4096)]
    display_rows.append(("attention", 8, 4096, "Global attention", cells))
    if report or not precision_cells:
        display_rows.extend(("stack", 1, n, "Two-layer decoder · FP16 experts", cells) for n in (64, 4096))
    if precision_cells:
        display_rows.extend(("stack", 1, n, "Two-layer decoder · FP32 expert arithmetic", precision_cells) for n in (64, 4096))
    for scope, batch, length, label, values in display_rows:
        case = f"{scope}-b{batch}-l{length}"
        vals = [f'<td>{values[case,m]:,.1f}</td>' if (case,m) in values else '<td class="na">—</td>'
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
            "@@PERF_FIGURE@@": figure, "@@DISPATCHES@@":escape(dispatch_note),
            "@@PRECISION@@":precision_panel(precision_note), "@@CHECKPOINTS@@":proofbox}
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
