import math
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors as pc
import matplotlib.colors as mcolors
from cmcrameri import cm
from cmap import Colormap
import re
import matplotlib.pyplot as plt

# local imports
from model.blood_pressure.hormone_influence import HormoneInfluence
from model.blood_pressure.file_handling import get_species_names_and_indices
from model.blood_pressure.utils import hill_pos, hill_neg

minutes_per_day = 1440
minutes_per_year = minutes_per_day * 365.25
cm = Colormap('crameri:romaO')

# font size settigns for generation of thesis plots vs. smaller plots for quick visualization during model development
thesis_plots = True
if thesis_plots:
    x_axis_title_font_size = 33
    x_axis_tick_font_size = 33
    y_axis_title_font_size = 33
    y_axis_tick_font_size = 33
    figure_title_font_size = 40
    legend_font_size = 33
else:
    x_axis_title_font_size = 16
    x_axis_tick_font_size = 14
    y_axis_title_font_size = 16
    y_axis_tick_font_size = 14
    figure_title_font_size = 25
    legend_font_size = 16

def create_simulation_results_visualisation_hormone_influence(species: list, simulation_results: pd.DataFrame, simulation_parameter_values:pd.DataFrame, hormone_influence: HormoneInfluence, gender:str, rsna_scenario:str, hormone_scenario:str):
    # Create a Plotly interactive plot of all species in the simulation results
    # the next two lines create a view accessible on localhost in the browser
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Scaled Hormone Profiles</b>", "<b>Model Species Trajectories</b>", "<b>Hormone-Modulated Parameter Dynamics</b>"))
    
    # Define top x-axis (days)
    max_minutes = int(simulation_results["t"].max())
    tickvals = np.arange(0, max_minutes+1, 10*minutes_per_day)   # 0, 1440, 2880, ...
    ticktext = [f"{v/minutes_per_day:g}" for v in tickvals] 
    
    x_values, e2_values, p4_values = create_hormone_profiles(hormone_influence, max_minutes, int(max_minutes/2000))  # 2000 points for the hormone profiles for smoother curves, adjust as needed

    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=e2_values,
            mode="lines",
            name="E2",
            xaxis="x",
            line=dict(width=4)
        ),
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=p4_values,
            mode="lines",
            name="P4",
            xaxis="x",
            line=dict(width=4)
        ),
        row=1,
        col=1
    )

    ### Plot all species simulations
    t_downsampled = downsample(simulation_results["t"].to_numpy(), max_points=100000)
    sim_array = np.vstack(simulation_results["simulation_results"].to_numpy())
    # downsample
    sim_array_downsampled = downsample(sim_array, max_points=100000)

    all_species = get_species_names_and_indices()

    if species:
        reduced_species = {s: all_species[s] for s in species if s in all_species}
    else:
        reduced_species = all_species

    species_names = list(reduced_species.keys())
    species_indices = list(reduced_species.values())

    colors = [mcolors.to_hex(cm(x)) for x in np.linspace(0, 1, len(species_names))]

    for name, idx, color in zip(species_names, species_indices, colors):
        fig.add_trace(
            go.Scattergl(
                x=t_downsampled,                       # downsampled time
                y=sim_array_downsampled[:, idx],       # fast NumPy slicing
                mode="lines",
                name=name,                      # legend label
                line=dict(color=color, width=4)
            ),
            row=2,
            col=1
        )

    # Let Plotly auto-generate ticks for the primary x-axis
    fig.update_yaxes(title_text="<b>Scaled Hormone Profiles</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Abundance</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Simulation Parameters</b>", row=3, col=1)

    fig.update_xaxes(
        tickfont=dict(size=x_axis_tick_font_size),
        title="<b>Time [days]</b>",
        side="bottom",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        domain=[0, 1],
    )
    fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(size=x_axis_title_font_size)))

    fig.update_yaxes(
        tickfont=dict(size=y_axis_tick_font_size),
        title_font=dict(size=y_axis_title_font_size),
    )

    n = len(simulation_parameter_values.columns)-1
    colors = [mcolors.to_hex(cm(x)) for x in np.linspace(0, 1, n)]

    # Plot the simulation parameter values
    for col,color in zip(simulation_parameter_values.columns,colors):
        if col == "t":
            continue
        fig.add_trace(
            go.Scattergl(
                x=simulation_parameter_values["t"],
                y=simulation_parameter_values[col],
                mode="lines",
                name=col,
                line=dict(
                    color=color,
                    width=4
                )
            ),
            row=3,
            col=1
        )

    fig.update_layout(
        title=dict(
            text=f"Blood Pressure Simulation for {gender}s ({rsna_scenario.upper()})<br> Hormone Scenario: {prettify_as_is(hormone_scenario)}",
            font=dict(size=figure_title_font_size, color="black"),
            x=0.5,  # center
            xanchor='center'
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )
    
    return fig

def create_MAP_comparison_visualisation(simulation_results: pd.DataFrame, baseline_MAP: float, gender:str, rsna_scenario:str):
    # Create a Plotly interactive plot of all species in the simulation results
    # the next two lines create a view accessible on localhost in the browser
    fig = go.Figure() 
    # Define top x-axis (days)
    max_minutes = int(simulation_results["t"].max())
    tickvals = np.arange(0, max_minutes+1, 10*minutes_per_day)   # 0, 1440, 2880, ...
    ticktext = [f"{v/minutes_per_day:g}" for v in tickvals] 

    ### Plot all MAP simulation results for the different scenarios
    n = len(simulation_results.columns)-1
    if n == 2:
        colors = [mcolors.to_hex(cm(0.24)), mcolors.to_hex(cm(0.73))]
    elif n == 3:
        colors = [mcolors.to_hex(cm(0.24)), mcolors.to_hex(cm(0.73)), mcolors.to_hex(cm(1))]
    else:
        colors = [mcolors.to_hex(cm(x)) for x in np.linspace(0.2, 0.99, n)]

    t_downsampled = downsample(simulation_results["t"].to_numpy())

    for col, color in zip(simulation_results.columns, colors):  # skip the 't' column
        if col == "t" or col == "baseline":
            continue
        sim_downsampled = downsample(simulation_results[col].to_numpy())
        fig.add_trace(
            go.Scattergl(
                x=t_downsampled,
                y=sim_downsampled,
                mode="lines",
                name=prettify_as_is(col),
                xaxis="x",
                legendgrouptitle=dict(text="Simulation Scenario", font=dict(size=25)),
                line=dict(
                    color=color,
                    width=4
                )
            )
        )

    # add a line for the null scenario as a reference
    fig.add_hline(y=baseline_MAP, line_dash="dash", line_color="gray")
    # add overall mean line description to legend
    fig.add_trace(
        go.Scattergl(
            x=[None],  # no actual data
            y=[None],
            mode="lines",
            line=dict(
                color="gray",
                dash="dash",
                width=4
            ),
            name="baseline steady state",
            showlegend=True
        ),
    )

    # Let Plotly auto-generate ticks for the primary x-axis
    fig.update_yaxes(title_text="<b>Mean Arterial Pressure [mmHg]</b>")

    fig.update_xaxes(
        tickfont=dict(size=x_axis_tick_font_size),
        title="<b>Time [days]</b>",
        side="bottom",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        domain=[0, 1],
    )
    fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(size=x_axis_title_font_size)))

    fig.update_yaxes(
        tickfont=dict(size=y_axis_tick_font_size),
        title_font=dict(size=y_axis_title_font_size),
    )

    fig.update_layout(
        title=dict(
            text=f"Blood Pressure Simulation for {gender}s ({rsna_scenario.upper()})",
            font=dict(size=figure_title_font_size, color="black"),
            x=0.5,  # center
            xanchor='center'
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )
    
    return fig

def create_MAP_comparison_visualisation_hormone_influence(simulation_results: pd.DataFrame, baseline_MAP: float, hormone_influence: HormoneInfluence, gender:str, rsna_scenario:str):
    # Create a Plotly interactive plot of all species in the simulation results
    # the next two lines create a view accessible on localhost in the browser
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("<b>Scaled Hormone Profiles</b>", "<b>Mean Arterial Pressure [mmHg]</b>"))
    # Define top x-axis (days)
    max_minutes = int(simulation_results["t"].max())
    tickvals = np.arange(0, max_minutes+1, 10*minutes_per_day)   # 0, 1440, 2880, ...
    ticktext = [f"{v/minutes_per_day:g}" for v in tickvals] 

    x_values, e2_values, p4_values = create_hormone_profiles(hormone_influence, max_minutes, int(max_minutes/2000))  # 2000 points for the hormone profiles for smoother curves, adjust as needed

    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=e2_values,
            mode="lines",
            name="E2",
            xaxis="x",
            line=dict(width=4)
        ),
        row=1,
        col=1
    )
    
    fig.add_trace(
        go.Scattergl(
            x=x_values,
            y=p4_values,
            mode="lines",
            name="P4",
            xaxis="x",
            line=dict(width=4)
        ),
        row=1,
        col=1
    )

    ### Plot all MAP simulation results for the different scenarios
    n = len(simulation_results.columns)-1
    if n == 2:
        colors = [mcolors.to_hex(cm(0.24)), mcolors.to_hex(cm(0.73))]
    elif n == 3:
        colors = [mcolors.to_hex(cm(0.24)), mcolors.to_hex(cm(0.73)), mcolors.to_hex(cm(1))]
    else:
        colors = [mcolors.to_hex(cm(x)) for x in np.linspace(0.2, 0.99, n)]

    t_downsampled = downsample(simulation_results["t"].to_numpy())

    for col, color in zip(simulation_results.columns, colors):  # skip the 't' column
        if col == "t" or col == "baseline":
            continue
        sim_downsampled = downsample(simulation_results[col].to_numpy())
        fig.add_trace(
            go.Scattergl(
                x=t_downsampled,
                y=sim_downsampled,
                mode="lines",
                name=prettify_as_is(col),
                xaxis="x",
                legendgrouptitle=dict(text="Simulation Scenario", font=dict(size=25)),
                line=dict(
                    color=color,
                    width=4
                )
            ),
            row=2,
            col=1
        )

    # add a line for the null scenario as a reference
    fig.add_hline(y=baseline_MAP, line_dash="dash", line_color="gray", row=2, col=1)
    # add overall mean line description to legend
    fig.add_trace(
        go.Scattergl(
            x=[None],  # no actual data
            y=[None],
            mode="lines",
            line=dict(
                color="gray",
                dash="dash",
                width=4
            ),
            name="baseline steady state",
            showlegend=True
        ),
        row=2, col=1
    )

    # Let Plotly auto-generate ticks for the primary x-axis
    fig.update_yaxes(title_text="<b>Scaled Hormone Profiles</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Mean Arterial Pressure [mmHg]</b>", row=2, col=1)

    fig.update_xaxes(
        tickfont=dict(size=x_axis_tick_font_size),
        title="<b>Time [days]</b>",
        side="bottom",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        domain=[0, 1],
    )
    fig.for_each_xaxis(lambda axis: axis.title.update(font=dict(size=x_axis_title_font_size)))

    fig.update_yaxes(
        tickfont=dict(size=y_axis_tick_font_size),
        title_font=dict(size=y_axis_title_font_size),
    )

    fig.update_layout(
        title=dict(
            text=f"Blood Pressure Simulation for {gender}s ({rsna_scenario.upper()})",
            font=dict(size=figure_title_font_size, color="black"),
            x=0.5,  # center
            xanchor='center'
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )
    
    return fig

def create_evaluation_result_plots(species: list, df_diff : pd.DataFrame, gender:str, rsna_scenario:str, hormone_scenario:str):
    # Create a Plotly interactive plot of all species differences in the simulation results
    fig = go.Figure()
    # Define top x-axis (days)
    max_minutes = df_diff["t"].max()
    tickvals = np.arange(0, max_minutes+1, 14400) 
    ticktext = [f"{v/1440:g}" for v in tickvals] 

    # select species to plot
    all_species = get_species_names_and_indices()
    if species:
        reduced_species = {s: all_species[s] for s in species if s in all_species}
    else:
        reduced_species = all_species

    n = len(reduced_species)
    colors = [mcolors.to_hex(cm(x)) for x in np.linspace(0, 1, n)]

    eval_downsampled = downsample_df(df_diff)
    t_downsampled = eval_downsampled["t"].to_numpy()


    for col, color in zip(df_diff.columns[1:], colors):  # skip the 't' column
        if col == "t" or col not in reduced_species:
            continue
        diff_downsampled = eval_downsampled[col].to_numpy()
        fig.add_trace(go.Scattergl(
            x=t_downsampled,
            y=diff_downsampled,
            mode="lines",
            name=col,
            xaxis="x",
            showlegend=True,
            legendgrouptitle=dict(text="Model Species", font=dict(size=25)),
            line=dict(
                    color=color,
                    width=4
                )
        ))

    fig.update_layout(
        title=dict(
            text=f"Relative Difference in {gender}s ({rsna_scenario.upper()})<br> Hormone Scenario: {prettify_as_is(hormone_scenario)}",
            x=0.5,
            xanchor="center",
            font=dict(size=figure_title_font_size),
        ),
        yaxis=dict(
            title="<b>Relative Difference from Baseline Average [%]</b>",
            title_font=dict(size=y_axis_title_font_size),
            tickfont=dict(size=y_axis_tick_font_size),
            ticks="outside"
        ),
        xaxis=dict(
            title="<b>Time [days]</b>",
            side="bottom",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            domain=[0, 1],
            tickfont=dict(size=x_axis_tick_font_size),
            title_font=dict(size=x_axis_title_font_size)
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
    )

    return fig

def create_simulation_average_plots(scenario_avgs:dict, gender:str, rsna_scenario:str):
    fig = go.Figure()

    baseline_pma = scenario_avgs['baseline']
    scenario_avgs.pop('baseline', None)
    x = list(scenario_avgs.keys())
    y = list(scenario_avgs.values())
    y_diff = [v - baseline_pma for v in y]

    # Pick color scales
    above_scale = pc.sequential.Redor
    below_scale = pc.sequential.Darkmint[1:]

    above = [val for val in y_diff if val >= 0]
    below = [val for val in y_diff if val < 0]

    # Normalize values for each scale
    def normalize(val, group):
        return (val - min(group)) / (max(group) - min(group)) if max(group) > min(group) else 0.5

    colors = []
    for val in y_diff:
        if val >= 0:
            frac = normalize(val, above)
            colors.append(pc.sample_colorscale(above_scale, frac)[0])
        else:
            frac = normalize(val, below)
            colors.append(pc.sample_colorscale(below_scale, frac)[0])


    hovertext = [f"Ø P_ma {k}: {v:.2f}" for k, v in scenario_avgs.items()]
    fig.add_trace(
        go.Bar(
            x=[prettify_as_is(str(s)) for s in x],
            y=y_diff,
            base=baseline_pma,
            hovertext=hovertext,
            name="MAP avg.",
            marker=dict(color=colors),
            showlegend=False
        )
    )

    # add a line for the null scenario as a reference
    fig.add_hline(y=baseline_pma, line_dash="dash", line_color="gray")
    # add overall mean line description to legend
    fig.add_trace(
        go.Scattergl(
            x=[None],  # no actual data
            y=[None],
            mode="lines",
            line=dict(
                color="gray",
                dash="dash",
                width=4
            ),
            name="baseline steady state",
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=["#9b9b9b"]),
            name="MAP avg.",
            showlegend=True
        )
    )

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=pc.sample_colorscale(pc.sequential.Redor, 0.7)[0], size=20, symbol="square", line=dict(width=0)),
        name="Above baseline",
        showlegend=True
    ))

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=pc.sample_colorscale(pc.sequential.Darkmint, 0.7)[0], size=20, symbol="square", line=dict(width=0)),
        name="Below baseline",
        showlegend=True
    ))

    for trace in fig.data:
        if isinstance(trace, go.Bar):
            trace.marker.line = dict(width=0.3, color="rgba(0,0,0,0.3)")

    fig.update_layout(
        title=dict(
            text=f"Average Mean Arterial Pressure in {gender}s ({rsna_scenario.upper()})",
            x=0.5,
            xanchor="center",
            font=dict(size=figure_title_font_size),
        ),
        yaxis=dict(
            title="<b>Mean Arterial Pressure [mmHg]</b>",
            title_font=dict(size=y_axis_title_font_size),
            tickfont=dict(size=y_axis_tick_font_size),
            range=[baseline_pma - 2, baseline_pma + 2],
            ticks="outside"
        ),
        xaxis=dict(
            tickangle=0,
            tickfont=dict(size=x_axis_tick_font_size),
            title_font=dict(size=x_axis_title_font_size),
            tickmode="array",
            tickvals=list(fig.data[0].x),
            ticktext=[f"<b>{wrap_label(x)}</b>" for x in list(fig.data[0].x)],
            ticks="outside"
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            traceorder="grouped",
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        barmode='group',
        bargap=0.2,
        bargroupgap=0.025,
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0, x1=1,
                y0=baseline_pma, y1=baseline_pma,
                line=dict(color="gray", width=4, dash="dash")
            )
        ]
    )
    
    return fig     

def create_simulation_average_plots_line(scenario_avgs:dict, gender:str, rsna_scenario:str):
    fig = go.Figure()

    baseline_pma = float(scenario_avgs["baseline"])
    scenario_data = {
        str(k): float(v)
        for k, v in scenario_avgs.items()
        if k != "baseline"
    }

    grouped = {
        "E2 + P4": [],
        "P4 only": [],
        "E2 only": [],
    }

    for scenario, value in scenario_data.items():
        group, x_strength, sort_key = parse_group_and_strength(scenario)
        if group not in grouped:
            continue

        grouped[group].append({
            "scenario": scenario,
            "x": x_strength,
            "y": value,
            "value": value,
            "label": prettify_as_is(str(scenario)),
            "sort_key": sort_key,
        })

    # color palette close to your original style
    group_colors = {
        "E2 + P4": pc.sample_colorscale(pc.sequential.Redor, 0.7)[0],
        "P4 only": pc.sample_colorscale(pc.sequential.Darkmint, 0.7)[0],
        "E2 only": pc.sample_colorscale(pc.sequential.Blues, 0.7)[0],
    }

    group_markers = {
        "E2 + P4": "circle",
        "P4 only": "square",
        "E2 only": "diamond",
    }

    x_tickvals = []
    x_ticktext = []

    for group_name, items in grouped.items():
        if not items:
            continue

        items.sort(key=lambda item: item["sort_key"])

        x_vals = [item["x"] for item in items]
        y_vals = [item["y"] for item in items]
        labels = [item["label"] for item in items]
        hovertext = [f"Ø P_ma {item['scenario']}: {item['value']:.2f}" for item in items]

        fig.add_trace(
            go.Scattergl(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=group_name,
                line=dict(color=group_colors[group_name], width=4),
                marker=dict(
                    size=10,
                    color=group_colors[group_name],
                    symbol=group_markers[group_name],
                    line=dict(width=0.4, color="rgba(0,0,0,0.35)")
                ),
                hovertext=hovertext,
                text=labels,
                customdata=[
                    [item["scenario"], item["value"]] for item in items
                ],
                showlegend=True
            )
        )

        x_tickvals.extend(x_vals)

    x_tickvals = sorted({float(v) for v in x_tickvals})
    x_ticktext = [f"<b>{v:g}</b>" for v in x_tickvals]

    # Add a reference baseline line
    fig.add_hline(y=baseline_pma, line_dash="dash", line_color="gray")

    fig.add_trace(
        go.Scattergl(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="gray", dash="dash", width=4),
            name="baseline steady state",
            showlegend=True
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Average Mean Arterial Pressure in {gender}s ({rsna_scenario.upper()})",
            x=0.5,
            xanchor="center",
            font=dict(size=figure_title_font_size),
        ),
        yaxis=dict(
            title="<b>Mean Arterial Pressure [mmHg]</b>",
            title_font=dict(size=y_axis_title_font_size),
            tickfont=dict(size=y_axis_tick_font_size),
            range=[baseline_pma - 2, baseline_pma + 2],
            ticks="outside"
        ),
        xaxis=dict(
            title="<b>Hormone strength</b>",
            tickangle=0,
            tickfont=dict(size=x_axis_tick_font_size),
            title_font=dict(size=x_axis_title_font_size),
            tickmode="array",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            ticks="outside"
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            traceorder="grouped",
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0, x1=1,
                y0=baseline_pma, y1=baseline_pma,
                line=dict(color="gray", width=4, dash="dash")
            )
        ]
    )

    return fig

def create_simulation_average_plots_per_cycle_phase(cycle_phase_data:pd.DataFrame, gender:str, rsna_scenario:str):
    fig = go.Figure()

    baseline_pma = cycle_phase_data.iat[0, 0] # baseline P_ma for comparisons
    df_diff = cycle_phase_data.subtract(cycle_phase_data['baseline'], axis=0)
    # rename indices to only show cycle phase names in brackets (shorter)
    df_diff.rename(index={key: re.search(r"\(([^)]+)\)", key).group(1) for key in df_diff.index}, inplace=True)

    df_diff = df_diff.drop(columns=['baseline'])
    cycle_phases = list(df_diff.index)
    # fixed anchor per phase (0–1)
    phase_anchor = dict(
        zip(
            cycle_phases,
            np.linspace(0.15, 0.85, len(cycle_phases))
        )
    )

    for phase in df_diff.index:
        y_vals = df_diff.loc[phase].values

        colors = [
            phase_locked_color(v, phase_anchor, phase)
            for v in y_vals
        ]

        fig.add_trace(
            go.Bar(
                x=[prettify_as_is(str(col)) for col in df_diff.columns],
                y=y_vals,
                base=baseline_pma,
                name=prettify_upper(phase),
                marker=dict(color=colors),
                showlegend=False
            )
        )

    phase_greys = {
        "FP":  "#cfcfcf",  # light grey, clearly visible on white
        "LP":  "#b5b5b5",  # light–mid grey
        "EFP": "#9b9b9b",  # mid grey
        "LFP": "#7f7f7f",  # dark grey
        "MLP": "#636363",  # very dark grey
    }
    
    for phase, col in phase_greys.items():
        fig.add_trace(
            go.Scattergl(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=col),
                name=phase,
                legendgroup="phase",
                legendgrouptitle=dict(text="Cycle Phase", font=dict(size=25)),
                showlegend=True
            )
        )

    # add overall mean line description to legend
    fig.add_trace(
        go.Scattergl(
            x=[None],  # no actual data
            y=[None],
            mode="lines",
            line=dict(
                color="gray",
                dash="dash",
                width=4
            ),
            name="baseline steady state",
            showlegend=True
        )
    )

    # add a line for the null scenario as a reference
    fig.add_hline(y=baseline_pma, line_dash="dash", line_color="gray")

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=pc.sample_colorscale(pc.sequential.Redor, 0.7)[0], size=20, symbol="square", line=dict(width=0)),
        name="Above baseline",
        showlegend=True
    ))

    fig.add_trace(go.Scattergl(
        x=[None], y=[None],
        mode="markers",
        marker=dict(color=pc.sample_colorscale(pc.sequential.Darkmint, 0.7)[0], size=20, symbol="square", line=dict(width=0)),
        name="Below baseline",
        showlegend=True
    ))

    for trace in fig.data:
        if isinstance(trace, go.Bar):
            trace.marker.line = dict(width=0.3, color="rgba(0,0,0,0.3)")

    fig.update_layout(
        title=dict(
            text=f"Mean Arterial Pressure in {gender}s ({rsna_scenario.upper()}) across menstrual cycle phases",
            x=0.5,
            xanchor="center",
            font=dict(size=figure_title_font_size),
        ),
        yaxis=dict(
            title="<b>Mean Arterial Pressure [mmHg]</b>",
            title_font=dict(size=y_axis_title_font_size),
            tickfont=dict(size=y_axis_tick_font_size),
            range=[baseline_pma - 1, baseline_pma + 1],
            ticks="outside"
        ),
        xaxis=dict(
            tickangle=0,
            tickfont=dict(size=x_axis_tick_font_size),
            title_font=dict(size=x_axis_title_font_size),
            tickmode="array",
            tickvals=list(fig.data[0].x),
            ticktext=[f"<b>{wrap_label(x)}</b>" for x in list(fig.data[0].x)],
            ticks="outside"
        ),
        legend=dict(
            font=dict(size=legend_font_size),
            traceorder="grouped",
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        ),
        barmode='group',
        bargap=0.2,
        bargroupgap=0.025,
        shapes=[
            dict(
                type="line",
                xref="paper",
                x0=0, x1=1,
                y0=baseline_pma, y1=baseline_pma,
                line=dict(color="gray", width=4, dash="dash")
            )
        ]
    )
    
    return fig

def create_hormone_profiles(hormone_influence:HormoneInfluence, max_minutes:int, plot_step:int=60):
    # include one more cycle after the hormone influence duration to show fading of hormone levels
    total_minutes = min((hormone_influence.duration + 1) * hormone_influence.cycle.cycle_length * minutes_per_day, max_minutes)
    plot_minutes = np.arange(0, total_minutes - hormone_influence.start_t, plot_step)
    plot_days = plot_minutes / minutes_per_day

    cycle_len_days = hormone_influence.cycle.cycle_length

    e2_after_transition = np.array([
        hormone_influence.cycle.get_scaled_estradiol_concentration(
            day % cycle_len_days)
        for day in plot_days
    ])

    p4_after_transition = np.array([
        hormone_influence.cycle.get_scaled_progesterone_concentration(
            day % cycle_len_days)
        for day in plot_days
    ])

    start_idx = int(hormone_influence.start_t // plot_step)

    total_plot_len = int(np.ceil(max_minutes / plot_step)) + 1

    e2_values = np.zeros(total_plot_len)
    p4_values = np.zeros(total_plot_len)

    end_idx = start_idx + len(e2_after_transition)

    e2_values[start_idx:end_idx] = e2_after_transition
    p4_values[start_idx:end_idx] = p4_after_transition

    # generate fading hormone levels after the influence duration ends
    fading_start_idx = int((hormone_influence.start_t + hormone_influence.duration * hormone_influence.cycle.cycle_length * minutes_per_day) // plot_step)
    fading_end_idx = end_idx # int(min(fading_start_idx + (hormone_influence.cycle.cycle_length * minutes_per_day)//plot_step, total_plot_len))
    
    fading_length = fading_end_idx - fading_start_idx
    hormone_potency = np.linspace(1.0, 0.0, num=fading_length) # hormone potency fading linearly
    e2_values[fading_start_idx:fading_end_idx] *= hormone_potency
    p4_values[fading_start_idx:fading_end_idx] *= hormone_potency
    
    # generate x values for the entire plot length
    x_values = np.arange(total_plot_len) * plot_step

    return x_values, e2_values, p4_values

def phase_locked_color(value, phase_anchor, phase):
    if value >= 0:
        return pc.sample_colorscale(
            pc.sequential.Redor,
            phase_anchor[phase]
        )[0]
    else:
        return pc.sample_colorscale(
            pc.sequential.Darkmint,
            phase_anchor[phase]
        )[0]

def create_hill_plots(hill_type:str="positive", exponents:list=[1], K:float=0.5, start:int=0, end:int=1, ts:list=[]):
    ts = np.arange(start, end, step=0.01) if not ts else ts
   
    fig = go.Figure()

    for ex in exponents:
        if hill_type == "positive":
            y = [hill_pos(t, ex, K) for t in ts]
        else:
            y = [hill_neg(t, ex, K) for t in ts]

        fig.add_trace(
            go.Scattergl(
                y=y,
                #x=ts,
                mode="lines",
                line=dict(
                    dash="dash",
                    width=4
                ),
                name=ex,
                showlegend=True
            )
        )

    fig.update_layout(
        title=dict(
            text=f"Competitive Binding Process for H{'-' if hill_type=='negative' else '+'}, K={K}",
            x=0.5,
            xanchor="center",
        )
    )

    return fig

def parse_group_and_strength(scenario: str):
        scenario_lower = scenario.lower()

        hormone_values = {}
        hormone_pattern = re.compile(r"(e2|p4)(?:[_\s+-]*)(\d+(?:\.\d+)?)?")

        for hormone, value in hormone_pattern.findall(scenario_lower):
            if value:
                hormone_values[hormone] = float(value)

        has_e2 = "e2" in scenario_lower
        has_p4 = "p4" in scenario_lower

        if has_e2 and has_p4:
            e2_strength = hormone_values.get("e2")
            p4_strength = hormone_values.get("p4")

            if e2_strength is None and p4_strength is not None:
                e2_strength = p4_strength
            if p4_strength is None and e2_strength is not None:
                p4_strength = e2_strength
            if e2_strength is None and p4_strength is None:
                numeric_values = re.findall(r"\d+(?:\.\d+)?", scenario_lower)
                if numeric_values:
                    e2_strength = p4_strength = float(numeric_values[0])

            e2_strength = e2_strength if e2_strength is not None else 0.0
            p4_strength = p4_strength if p4_strength is not None else 0.0
            return "E2 + P4", e2_strength, (e2_strength, p4_strength)
        elif has_p4:
            p4_strength = hormone_values.get("p4")
            if p4_strength is None:
                numeric_values = re.findall(r"\d+(?:\.\d+)?", scenario_lower)
                p4_strength = float(numeric_values[0]) if numeric_values else 0.0
            return "P4 only", p4_strength, (p4_strength,)
        elif has_e2:
            e2_strength = hormone_values.get("e2")
            if e2_strength is None:
                numeric_values = re.findall(r"\d+(?:\.\d+)?", scenario_lower)
                e2_strength = float(numeric_values[0]) if numeric_values else 0.0
            return "E2 only", e2_strength, (e2_strength,)
        else:
            return "Other", 0.0, (float("inf"),)
    
def prettify_upper(name: str) -> str:
    return name.replace("_", " ").upper()

def prettify_title(name: str) -> str:
    return name.replace("_", " ").title()

def prettify_as_is(name: str) -> str:
    return name.replace("_", " ")

def wrap_label(s):
    #return s.replace(" on ", "<br> on <br>").replace(" On ", "<br> on <br>").replace(" ON ", "<br> on <br>").replace("Secretion", "Secr.")
    return s.replace("E2 on ", "").replace("E2 On ", "").replace("E2 ON ", "").replace("P4 on ", "").replace("P4 On ", "").replace("P4 ON ", "").replace("Secretion", "Secr.").replace("Renin ", "Renin<br>")

def downsample(data, max_points=4000):
    n = len(data)
    if n <= max_points:
        return data
    step = math.ceil(n / max_points)
    return data[::step]

def downsample_df(df:pd.DataFrame, max_points=4000):
    n = len(df)
    if n <= max_points:
        return df
    step = (n + max_points - 1) // max_points
    return df.iloc[::step]
