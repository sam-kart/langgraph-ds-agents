import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from graph.state import AgentState
from tools.llm import complete

_DEFAULT_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _outputs_dir() -> str:
    return os.path.join(os.environ.get("DS_OUTPUTS_DIR", _DEFAULT_OUTPUTS), "outputs")


SYSTEM_PROMPT = """You are a Data Visualization Lead building an executive dashboard for a data science project.

Your audience is a hiring manager, ML engineer, product stakeholder, or technical reviewer who wants to understand the run quickly without reading every log. Your job is not to decorate; your job is to communicate.

Return ONLY valid JSON with this structure:
{
  "title": "short dashboard title",
  "subtitle": "one sentence explaining the run",
  "headline": "one sentence with the most important result",
  "metric_cards": [
    {"label": "Best model", "value": "...", "context": "why it matters"},
    {"label": "Primary metric", "value": "...", "context": "test set or CV context"},
    {"label": "Review status", "value": "...", "context": "highest severity issue"}
  ],
  "caveats": ["specific limitation or methodology warning"],
  "next_steps": ["specific next action"],
  "plot_captions": {"filename.png": "one sentence explaining what to look for"}
}

Dashboard quality rules:
- Optimize for scanning: concise text, no prose walls.
- Use exact numbers from the report or logs when available.
- Do not invent metrics. If a value is unavailable, say "Not reported".
- Explain whether a model improvement is meaningful or within noise.
- Mention HIGH reviewer findings prominently.
- Captions should interpret the visual, not repeat the filename.
- Keep language professional and resume-demo friendly.
"""


def _strip_json_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _latest_report(outputs_dir: str, report_path: str) -> str:
    if report_path and os.path.exists(report_path):
        return report_path
    reports = sorted(Path(outputs_dir).glob("report_*.md"))
    return str(reports[-1]) if reports else ""


def _read_text(path: str, limit: int = 12000) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        text = f.read()
    return text[:limit]


def _csv_profile(outputs_dir: str) -> list[dict]:
    profiles = []
    for name in ("clean_data.csv", "engineered_data.csv"):
        path = os.path.join(outputs_dir, name)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            profiles.append(
                {
                    "name": name,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "preview_columns": list(df.columns[:10]),
                }
            )
        except Exception as exc:
            profiles.append({"name": name, "error": str(exc)})
    return profiles


def _fallback_content(problem: str, plots: list[str], report_text: str) -> dict:
    first_line = next((line.strip("# ").strip() for line in report_text.splitlines() if line.strip()), "Data Science Run")
    return {
        "title": first_line or "Data Science Run Dashboard",
        "subtitle": problem,
        "headline": "Review the report, model outputs, and visual artifacts from this run.",
        "metric_cards": [
            {"label": "Report", "value": "Generated", "context": "Markdown report is linked below."},
            {"label": "Plots", "value": str(len(plots)), "context": "Visual artifacts available in this dashboard."},
            {"label": "Review", "value": "See caveats", "context": "Reviewer findings are summarized in the report."},
        ],
        "caveats": ["Dashboard text fell back to deterministic content because LLM JSON parsing failed."],
        "next_steps": ["Open the full report for exact methodology and metrics."],
        "plot_captions": {plot: "Generated visual artifact from the analysis run." for plot in plots},
    }


def _dashboard_content(state: AgentState, outputs_dir: str, plots: list[str], report_text: str, profiles: list[dict]) -> dict:
    review_text = "\n\n".join(state.get("reviews", []))[-5000:]
    history_tail = []
    for entry in state.get("code_history", [])[-4:]:
        history_tail.append(
            {
                "agent": entry.get("agent", "agent"),
                "task": entry.get("task", "")[:500],
                "output_tail": (entry.get("output") or entry.get("error") or "")[-1200:],
            }
        )

    prompt = json.dumps(
        {
            "problem": state["problem"],
            "report_excerpt": report_text[-9000:],
            "reviews": review_text,
            "recent_agent_outputs": history_tail,
            "plots": plots,
            "data_profiles": profiles,
        },
        indent=2,
    )

    try:
        raw = complete(SYSTEM_PROMPT, prompt, max_tokens=4096)
        return json.loads(_strip_json_fences(raw))
    except Exception:
        return _fallback_content(state["problem"], plots, report_text)


def _safe_list(items: list[str]) -> str:
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        items = []
    if not items:
        return "<li>Not reported.</li>"
    return "\n".join(f"<li>{html.escape(str(item))}</li>" for item in items)


def _safe_cards(cards) -> list[dict]:
    if isinstance(cards, dict):
        cards = [cards]
    if not isinstance(cards, list):
        return []
    return [card for card in cards if isinstance(card, dict)][:6]


def _render_dashboard(
    state: AgentState,
    content: dict,
    outputs_dir: str,
    plots: list[str],
    profiles: list[dict],
    report_path: str,
) -> str:
    cards = _safe_cards(content.get("metric_cards", []))
    card_html = "\n".join(
        f"""
        <article class="metric-card">
          <div class="metric-label">{html.escape(str(card.get("label", "Metric")))}</div>
          <div class="metric-value">{html.escape(str(card.get("value", "Not reported")))}</div>
          <p>{html.escape(str(card.get("context", "")))}</p>
        </article>
        """
        for card in cards
    )

    profile_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(str(profile.get("name", "")))}</td>
          <td>{html.escape(str(profile.get("rows", profile.get("error", "N/A"))))}</td>
          <td>{html.escape(str(profile.get("columns", "N/A")))}</td>
          <td>{html.escape(", ".join(profile.get("preview_columns", [])))}</td>
        </tr>
        """
        for profile in profiles
    ) or '<tr><td colspan="4">No CSV artifacts found.</td></tr>'

    captions = content.get("plot_captions", {}) or {}
    plot_html = "\n".join(
        f"""
        <figure class="plot-card">
          <img src="{html.escape(plot)}" alt="{html.escape(captions.get(plot, plot))}">
          <figcaption>
            <strong>{html.escape(plot)}</strong>
            <span>{html.escape(captions.get(plot, "Generated visual artifact from the analysis run."))}</span>
          </figcaption>
        </figure>
        """
        for plot in plots
    ) or '<p class="muted">No PNG visualizations were generated for this run.</p>'

    report_name = os.path.basename(report_path) if report_path else ""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(content.get("title", "Data Science Dashboard")))}</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --surface: #ffffff;
      --ink: #17202a;
      --muted: #5d6a78;
      --line: #d9dee5;
      --accent: #2266aa;
      --accent-2: #147d64;
      --warning: #9a5b00;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      background: #0f1720;
      color: white;
      padding: 32px 24px;
      border-bottom: 4px solid var(--accent-2);
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto 48px;
    }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: clamp(28px, 4vw, 44px); line-height: 1.1; max-width: 980px; }}
    h2 {{ font-size: 22px; margin-bottom: 14px; }}
    p {{ margin: 0; }}
    .subtitle {{ color: #d3dbe5; max-width: 900px; margin-top: 10px; font-size: 17px; }}
    .meta {{ color: #b6c2cf; margin-top: 16px; font-size: 14px; }}
    .section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
      margin-bottom: 18px;
    }}
    .headline {{
      border-left: 5px solid var(--accent);
      padding-left: 16px;
      font-size: 18px;
      color: #243241;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fbfcfd;
      min-height: 138px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .metric-value {{
      color: var(--accent);
      font-size: 26px;
      font-weight: 750;
      margin-top: 6px;
      overflow-wrap: anywhere;
    }}
    .metric-card p {{ color: var(--muted); margin-top: 8px; font-size: 14px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef2f6; color: #263442; }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .plot-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: white;
    }}
    .plot-card img {{
      width: 100%;
      height: auto;
      display: block;
      background: white;
    }}
    figcaption {{
      border-top: 1px solid var(--line);
      padding: 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    figcaption strong {{
      display: block;
      color: var(--ink);
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 7px 0; }}
    .two-col {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .muted {{ color: var(--muted); }}
    .link-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .button-link {{
      color: white;
      background: var(--accent);
      text-decoration: none;
      padding: 9px 12px;
      border-radius: 6px;
      font-weight: 650;
    }}
    .button-link.secondary {{ background: var(--accent-2); }}
    @media (max-width: 640px) {{
      header {{ padding: 24px 16px; }}
      main {{ width: calc(100% - 20px); margin-top: 14px; }}
      .section {{ padding: 15px; }}
      .plot-grid {{ grid-template-columns: 1fr; }}
      .metric-value {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(str(content.get("title", "Data Science Dashboard")))}</h1>
    <p class="subtitle">{html.escape(str(content.get("subtitle", state["problem"])))}</p>
    <p class="meta">Generated {html.escape(generated)} · Static dashboard artifact</p>
  </header>
  <main>
    <section class="section">
      <h2>Executive Readout</h2>
      <p class="headline">{html.escape(str(content.get("headline", "Review the run artifacts for results.")))}</p>
      <div class="link-row">
        {f'<a class="button-link" href="{html.escape(report_name)}">Open Full Report</a>' if report_name else ''}
        <a class="button-link secondary" href="clean_data.csv">Clean Data</a>
        <a class="button-link secondary" href="engineered_data.csv">Engineered Data</a>
      </div>
    </section>

    <section class="section">
      <h2>Key Metrics</h2>
      <div class="metric-grid">{card_html}</div>
    </section>

    <section class="section">
      <h2>Data Artifacts</h2>
      <table>
        <thead><tr><th>File</th><th>Rows</th><th>Columns</th><th>Preview Columns</th></tr></thead>
        <tbody>{profile_rows}</tbody>
      </table>
    </section>

    <section class="section">
      <h2>Visual Evidence</h2>
      <div class="plot-grid">{plot_html}</div>
    </section>

    <section class="section">
      <div class="two-col">
        <div>
          <h2>Caveats</h2>
          <ul>{_safe_list(content.get("caveats", []))}</ul>
        </div>
        <div>
          <h2>Recommended Next Steps</h2>
          <ul>{_safe_list(content.get("next_steps", []))}</ul>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def dashboarder_node(state: AgentState) -> dict:
    outputs_dir = _outputs_dir()
    os.makedirs(outputs_dir, exist_ok=True)

    report_path = _latest_report(outputs_dir, state.get("report_path", ""))
    report_text = _read_text(report_path)
    plots = sorted(name for name in os.listdir(outputs_dir) if name.endswith(".png"))
    profiles = _csv_profile(outputs_dir)
    content = _dashboard_content(state, outputs_dir, plots, report_text, profiles)

    html_text = _render_dashboard(state, content, outputs_dir, plots, profiles, report_path)
    dashboard_path = os.path.join(outputs_dir, "dashboard.html")
    with open(dashboard_path, "w") as f:
        f.write(html_text)

    return {
        "dashboard_path": dashboard_path,
        "final_summary": f"Dashboard saved to {dashboard_path}",
    }
