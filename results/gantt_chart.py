import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates
from datetime import date
import numpy as np

# ---------------------------------------------------------------------------
# Data definition
# ---------------------------------------------------------------------------

# Phases: (label, hex colour)
PHASES = {
    "Phase 1: Project Setup":           "#4C72B0",
    "Phase 2: Literature & Planning":   "#55A868",
    "Phase 3: Preliminary Report":      "#C44E52",
    "Phase 4: Full Experiments":        "#DD8452",
    "Phase 5: Analysis & Writing":      "#8172B2",
    "Phase 6: Final Submission":        "#937860",
}

# Tasks: (label, phase key, start date, end date)
TASKS = [
    # Phase 1
    ("Topic selection and allocation",           "Phase 1: Project Setup",          date(2025, 11,  1), date(2025, 12, 31)),
    ("Initial research and reading",             "Phase 1: Project Setup",          date(2026,  1,  1), date(2026,  1, 31)),
    # Phase 2
    ("Literature review",                        "Phase 2: Literature & Planning",  date(2026,  1,  1), date(2026,  2, 28)),
    ("Research question & methodology design",   "Phase 2: Literature & Planning",  date(2026,  2,  1), date(2026,  3, 31)),
    # Phase 3
    ("Data preprocessing and pipeline setup",    "Phase 3: Preliminary Report",     date(2026,  3,  1), date(2026,  4, 15)),
    ("MCAR experiments \u2014 all strategies",   "Phase 3: Preliminary Report",     date(2026,  4,  1), date(2026,  4, 25)),
    ("Feature-dependent missingness 30%",        "Phase 3: Preliminary Report",     date(2026,  4, 10), date(2026,  4, 25)),
    ("Preliminary report write-up",              "Phase 3: Preliminary Report",     date(2026,  4, 10), date(2026,  4, 30)),
    # Phase 4
    ("Feature-dependent missingness 10/20/30%",  "Phase 4: Full Experiments",       date(2026,  5,  1), date(2026,  5, 25)),
    ("Label scarcity experiments 5/10/20% labels","Phase 4: Full Experiments",      date(2026,  5,  1), date(2026,  5, 31)),
    ("Larger dataset search and setup",          "Phase 4: Full Experiments",       date(2026,  5,  1), date(2026,  5, 25)),
    ("Full experiments on larger dataset",       "Phase 4: Full Experiments",       date(2026,  5, 15), date(2026,  6, 20)),
    ("Results collection and validation",        "Phase 4: Full Experiments",       date(2026,  6,  1), date(2026,  6, 30)),
    # Phase 5
    ("Results analysis and interpretation",      "Phase 5: Analysis & Writing",     date(2026,  6, 15), date(2026,  7, 15)),
    ("Thesis writing \u2013 results and discussion","Phase 5: Analysis & Writing",  date(2026,  7,  1), date(2026,  7, 31)),
    # Phase 6
    ("Thesis polishing and proofreading",        "Phase 6: Final Submission",       date(2026,  7, 15), date(2026,  8,  5)),
    ("Code repository cleanup",                  "Phase 6: Final Submission",       date(2026,  7, 25), date(2026,  8,  5)),
    ("Video presentation recording",             "Phase 6: Final Submission",       date(2026,  7, 28), date(2026,  8,  5)),
]

# Milestones: (label, date)
MILESTONES = [
    ("Progress slides 1",              date(2026,  2,  5)),
    ("Progress slides 2",              date(2026,  3,  5)),
    ("Progress slides 3",              date(2026,  4,  2)),
    ("Preliminary report submission",  date(2026,  4, 30)),
    ("4th group supervision",          date(2026,  6, 26)),
    ("Progress slides 4",             date(2026,  7,  2)),
    ("2nd individual meeting",         date(2026,  7, 17)),
    ("Final submission",               date(2026,  8,  6)),
]

# ---------------------------------------------------------------------------
# Build chart
# ---------------------------------------------------------------------------

n_tasks = len(TASKS)
fig_height = max(11, n_tasks * 0.62 + 3.5)
fig, ax = plt.subplots(figsize=(14, fig_height))

# Draw task bars (first task at the bottom, last at the top after invert)
y_positions = list(range(n_tasks - 1, -1, -1))  # n_tasks-1 down to 0

for idx, (label, phase, start, end) in enumerate(TASKS):
    y = y_positions[idx]
    colour = PHASES[phase]
    start_num = mdates.date2num(start)
    end_num   = mdates.date2num(end)
    ax.barh(y, end_num - start_num, left=start_num,
            height=0.55, color=colour, alpha=0.88,
            linewidth=0.6, edgecolor="white")

# Map each milestone to the task row whose bar spans or is closest
def task_row_for_milestone(mdate):
    for idx, (label, phase, start, end) in enumerate(TASKS):
        if start <= mdate <= end:
            return y_positions[idx]
    best_idx = 0
    best_delta = float("inf")
    for idx, (label, phase, start, end) in enumerate(TASKS):
        delta = abs((mdate - end).days)
        if delta < best_delta:
            best_delta = delta
            best_idx = idx
    return y_positions[best_idx]

milestone_colour = "#1a1a1a"
for m_label, m_date in MILESTONES:
    y = task_row_for_milestone(m_date)
    ax.plot(mdates.date2num(m_date), y, marker="D",
            markersize=8, color=milestone_colour,
            markeredgecolor="white", markeredgewidth=0.8,
            zorder=5, clip_on=False)
    ax.annotate(m_label,
                xy=(mdates.date2num(m_date), y + 0.33),
                fontsize=7.0, ha="center", va="bottom",
                color="#1a1a1a",
                rotation=50,
                annotation_clip=False)

# ---------------------------------------------------------------------------
# Axes formatting
# ---------------------------------------------------------------------------
ax.set_yticks(list(range(n_tasks)))
ax.set_yticklabels([t[0] for t in reversed(TASKS)], fontsize=8.5)

ax.xaxis.set_major_locator(MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%b '%y"))
ax.set_xlim(mdates.date2num(date(2025, 11, 1)),
            mdates.date2num(date(2026,  8, 31)))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)

ax.set_ylim(-1.0, n_tasks - 0.4)
ax.invert_yaxis()

# Light vertical grid at each month
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#bbbbbb")
ax.set_axisbelow(True)
ax.yaxis.grid(False)

# Spine styling
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
phase_patches = [
    mpatches.Patch(facecolor=colour, alpha=0.88, label=phase)
    for phase, colour in PHASES.items()
]
milestone_handle = plt.Line2D(
    [0], [0], marker="D", color="w",
    markerfacecolor=milestone_colour, markeredgecolor="white",
    markersize=8, label="Milestone"
)
ax.legend(
    handles=phase_patches + [milestone_handle],
    loc="lower right",
    fontsize=7.5,
    framealpha=0.9,
    edgecolor="#cccccc",
    ncol=2,
)

# ---------------------------------------------------------------------------
# Title and layout
# ---------------------------------------------------------------------------
ax.set_title("MSc Thesis Project Timeline  |  Nov 2025 \u2013 Aug 2026",
             fontsize=12, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=9, labelpad=6)

fig.tight_layout(pad=1.4)
fig.savefig("results/gantt_chart.png", dpi=200, bbox_inches="tight")
print("Saved results/gantt_chart.png")
