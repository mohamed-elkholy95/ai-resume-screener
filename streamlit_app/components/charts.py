"""Plotly chart helpers for AI Resume Screener dashboard."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Brand colors
COLOR_STRONG = "#2ecc71"
COLOR_MODERATE = "#f39c12"
COLOR_WEAK = "#e74c3c"
COLOR_BG = "#0e1117"
COLOR_CARD = "#262730"
COLOR_TEXT = "#fafafa"
COLOR_GRID = "#3d3d3d"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_CARD,
    font=dict(color=COLOR_TEXT, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
)


def _apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**PLOTLY_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


def plot_score_distribution(scores: list[float]) -> go.Figure:
    """Histogram of match scores with Strong/Moderate/Weak threshold markers."""
    fig = go.Figure()

    if scores:
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            marker=dict(
                color=scores,
                colorscale=[
                    [0.0, COLOR_WEAK],
                    [0.5, COLOR_MODERATE],
                    [1.0, COLOR_STRONG],
                ],
                cmin=0,
                cmax=100,
                line=dict(color=COLOR_BG, width=0.5),
            ),
            name="Candidates",
            hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
        ))

    # Threshold lines
    for val, label, color in [
        (50, "Moderate", COLOR_MODERATE),
        (75, "Strong", COLOR_STRONG),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_color=color,
        )

    fig.update_xaxes(title_text="Match Score (%)", range=[0, 100])
    fig.update_yaxes(title_text="Number of Candidates")
    return _apply_layout(fig, title="Score Distribution")


def plot_skills_coverage(skill_coverage: dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing % of resumes that contain each required skill."""
    if not skill_coverage:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLOR_TEXT, size=14))
        return _apply_layout(fig, title="Skills Coverage")

    skills = list(skill_coverage.keys())
    values = [skill_coverage[s] * 100 if skill_coverage[s] <= 1 else skill_coverage[s]
              for s in skills]

    colors_list = [
        COLOR_STRONG if v >= 75 else COLOR_MODERATE if v >= 50 else COLOR_WEAK
        for v in values
    ]

    # Sort by value descending
    sorted_pairs = sorted(zip(skills, values, colors_list), key=lambda x: x[1])
    skills, values, colors_list = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    fig = go.Figure(go.Bar(
        x=list(values),
        y=list(skills),
        orientation="h",
        marker=dict(color=list(colors_list)),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    fig.update_xaxes(title_text="Coverage (%)", range=[0, 100])
    fig.update_yaxes(title_text="Required Skill")
    return _apply_layout(fig, title="Skills Coverage Across Resumes",
                         height=max(300, len(skills) * 28 + 80))


def plot_experience_distribution(experiences: list[float]) -> go.Figure:
    """Bar chart bucketing candidates by years of experience."""
    fig = go.Figure()

    if experiences:
        # Bucket into ranges
        buckets = {"0–2 yrs": 0, "2–5 yrs": 0, "5–8 yrs": 0, "8–12 yrs": 0, "12+ yrs": 0}
        for exp in experiences:
            if exp is None:
                continue
            if exp < 2:
                buckets["0–2 yrs"] += 1
            elif exp < 5:
                buckets["2–5 yrs"] += 1
            elif exp < 8:
                buckets["5–8 yrs"] += 1
            elif exp < 12:
                buckets["8–12 yrs"] += 1
            else:
                buckets["12+ yrs"] += 1

        fig.add_trace(go.Bar(
            x=list(buckets.keys()),
            y=list(buckets.values()),
            marker=dict(
                color=list(buckets.values()),
                colorscale=[[0, COLOR_WEAK], [0.5, COLOR_MODERATE], [1, COLOR_STRONG]],
                line=dict(color=COLOR_BG, width=0.5),
            ),
            hovertemplate="%{x}: %{y} candidate(s)<extra></extra>",
        ))

    fig.update_xaxes(title_text="Experience Range")
    fig.update_yaxes(title_text="Number of Candidates")
    return _apply_layout(fig, title="Experience Distribution")


def plot_skills_gap(missing_skills: dict[str, int]) -> go.Figure:
    """Bar chart of the most commonly missing required skills."""
    if not missing_skills:
        fig = go.Figure()
        fig.add_annotation(text="No skills gap data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLOR_TEXT, size=14))
        return _apply_layout(fig, title="Skills Gap Analysis")

    sorted_skills = sorted(missing_skills.items(), key=lambda x: x[1], reverse=True)[:15]
    skills, counts = zip(*sorted_skills)

    fig = go.Figure(go.Bar(
        x=list(counts),
        y=list(skills),
        orientation="h",
        marker=dict(color=COLOR_WEAK),
        hovertemplate="%{y}: missing in %{x} resume(s)<extra></extra>",
    ))

    fig.update_xaxes(title_text="Number of Resumes Missing This Skill")
    fig.update_yaxes(title_text="Skill")
    return _apply_layout(fig, title="Top Missing Skills (Skills Gap)",
                         height=max(300, len(skills) * 30 + 80))


def plot_score_breakdown(breakdown: dict[str, float]) -> go.Figure:
    """Radar / spider chart of score components for a single candidate."""
    if not breakdown:
        fig = go.Figure()
        fig.add_annotation(text="No breakdown available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLOR_TEXT, size=14))
        return _apply_layout(fig, title="Score Breakdown")

    categories = list(breakdown.keys())
    values = [breakdown[k] for k in categories]

    # Close the radar loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=f"rgba(46,204,113,0.25)",
        line=dict(color=COLOR_STRONG, width=2),
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=COLOR_TEXT),
                gridcolor=COLOR_GRID,
            ),
            angularaxis=dict(tickfont=dict(color=COLOR_TEXT)),
            bgcolor=COLOR_CARD,
        ),
    )
    return _apply_layout(fig, title="Score Breakdown")


def plot_comparison(candidate_scores: dict[str, dict[str, float]]) -> go.Figure:
    """Grouped bar chart comparing multiple candidates across score dimensions."""
    if not candidate_scores:
        fig = go.Figure()
        fig.add_annotation(text="No candidates to compare", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=COLOR_TEXT, size=14))
        return _apply_layout(fig, title="Candidate Comparison")

    # Collect all dimensions
    all_dims: set[str] = set()
    for scores in candidate_scores.values():
        all_dims.update(scores.keys())
    dimensions = sorted(all_dims)

    palette = [COLOR_STRONG, COLOR_MODERATE, COLOR_WEAK,
               "#3498db", "#9b59b6", "#1abc9c", "#e67e22"]

    fig = go.Figure()
    for i, (name, scores) in enumerate(candidate_scores.items()):
        fig.add_trace(go.Bar(
            name=name[:20],
            x=dimensions,
            y=[scores.get(d, 0) for d in dimensions],
            marker_color=palette[i % len(palette)],
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Score Dimension")
    fig.update_yaxes(title_text="Score (%)", range=[0, 100])
    return _apply_layout(fig, title="Candidate Comparison")
