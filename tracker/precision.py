"""Display the separate precision-control acquisition only after it passes the device gates."""
from html import escape
import json
from statistics import median


def precision_panel(results):
    out = results / "precision-control-01"
    intro = ('<h3 class="subhead">Two-layer precision control</h3><p class="note">'
             'Both layers use FP32 for the expert input, projections, activation and weighted sum. '
             'The weights and persistent state stay in FP16. This workload uses a custom expert projection kernel.</p>')
    if not out.exists():
        return intro + '<p>Awaiting GPU correctness checks and synchronized timings.</p>'
    try:
        from bench.protocol import STACK, SCHEDULES
        from diagnostics.precision_study import load_verified
        runs, cohort = load_verified(out)
    except (OSError, KeyError, TypeError, ValueError, ImportError):
        states = []
        for session in ("a", "b"):
            try:
                data = json.loads((out / session / "manifest.json").read_text())
                states.append(f"Session {session.upper()} is {data['status']}.")
            except (KeyError, OSError, TypeError, ValueError):
                continue
        return intro + '<p>' + escape(" ".join(states + ["A verified two-session report is pending."])) + '</p>'
    rows = []
    for case in STACK:
        cells = [median(value for session in runs.values() for value in session[case.name][:, i])
                 for i in range(len(SCHEDULES))]
        rows.append(f"<tr><td>{case.length:,}</td>" + "".join(f"<td>{v:,.1f}</td>" for v in cells) + "</tr>")
    return (intro + '<p>Median synchronized latency in µs across two verified sessions.</p>'
            '<div class="scroll"><table class="results"><thead><tr><th>Visible tokens</th><th>MLX</th>'
            '<th>Staged</th><th>Hybrid</th><th>Single</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div><p class="note">Control source SHA-256 '
            + escape(cohort["control_sha256"]) + '.</p>')
