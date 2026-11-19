"""Derive control medians and figure provenance from the certified raw trials."""
import hashlib
from html import escape
import json
from statistics import median


def precision_results(results):
    out = results / "precision-control-01"
    if not out.exists():
        return {}, {}, "Awaiting GPU correctness checks and synchronized timings."
    try:
        from bench.protocol import STACK, SCHEDULES
        from diagnostics.precision_study import load_verified
        runs, cohort = load_verified(out)
        cells = {(case.name, method): median(value for session in runs.values()
                                             for value in session[case.name][:, i])
                 for case in STACK for i, method in enumerate(SCHEDULES)}
        provenance = {"source_sha256": cohort["machine"]["sources_sha256"],
                      "control_sha256": cohort["control_sha256"], "control": cohort["control"],
                      "control_sessions": sorted(runs), "control_case_sha256": {
                          session: {case.name: hashlib.sha256((out / session / (case.name+".csv")).read_bytes()).hexdigest()
                                    for case in STACK} for session in runs}}
        note = (f"Sessions A and B passed the 15 registered output and state checks against this control’s CPU reference "
                f"before and after timing. Both lengths have {len(runs) * cohort['trials']} paired blocks.")
        return cells, provenance, note
    except (OSError, KeyError, TypeError, ValueError, ImportError):
        states = []
        for session in ("a", "b"):
            try:
                data = json.loads((out / session / "manifest.json").read_text())
                states.append(f"Session {session.upper()} is {data['status']}.")
            except (KeyError, OSError, TypeError, ValueError):
                continue
        return {}, {}, " ".join(states + ["A verified two-session report is pending."])


def precision_panel(note):
    return ('<h3 class="subhead">Two-layer precision control</h3><p>'
            'The decoder keeps FP16 expert weights and computes the expert inputs, projections, '
            'activation and weighted sum in FP32 using a custom projection kernel. '
            'The isolated split-context control was measured only on global attention.</p>'
            f'<p class="note">{escape(note)}</p>')
