import numpy as np
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.stats import gaussian_kde


def summarize_assignments(gdf_res, distances, category=None):
    avg_distance_m = np.nanmean(distances)
    median_distance_m = np.nanmedian(distances)
    max_distance_m = np.nanmax(distances)
    if category is not None:
        category = f"({category})"
    else:
        category = ""
    display(HTML(
        f"<li><u><b>{category}</b></u>, Average distance: <b>{avg_distance_m:,.1f} m ({avg_distance_m/1000:,.2f} km)</b>, "
        f"Median: <b>{median_distance_m:,.1f} m ({median_distance_m/1000:,.2f} km)</b>, "
        f"Max: <b>{max_distance_m:,.1f} m ({max_distance_m/1000:,.2f} km)</b>, "
        f"# of nooddrinkwaterpunten: <b>{len(gdf_res.assigned_parking_lot.unique())}</b>. "
        f"% of residents within 1 km (or loopafstand): <b>{np.sum(distances <= 1000)/len(distances)*100:,.1f}%</b></li>"
    ))


def get_np_bins(distances_random, distances_optimised):
    x_max = max(distances_random.max(), distances_optimised.max())
    x_min = min(distances_random.min(), distances_optimised.min())
    np_bins = np.linspace(x_min, x_max, 50)
    return np_bins


def visualize_distances(fig, distances, name, np_bins, color=None):
    xbins = {
        "start": np_bins[0],
        "end": np_bins[-1],
        "size": np_bins[1] - np_bins[0]
    }
    # Assign color if not provided (cycle through Plotly palette)
    if color is None:
        palette = pc.qualitative.Plotly
        color = palette[len(fig.data) % len(palette)]

    # Histogram normalized to probability
    fig.add_trace(go.Histogram(
        x=distances,
        xbins=xbins,
        name=name,
        opacity=0.8,
        histnorm='probability',
        marker_color=color
    ))

    # Compute KDE
    kde = gaussian_kde(distances)
    y_kde = kde(np_bins)

    # Scale KDE to match histogram normalization
    bin_width = ((xbins['end'] - xbins['start']) / xbins['size'])
    y_kde_scaled = y_kde * bin_width  # scale density to probability per bin

    fig.add_trace(go.Scatter(
        x=np_bins,
        y=y_kde_scaled,
        mode='lines',
        name=f"{name} (Smoothed)",
        opacity=0.5,
        line=dict(color=color, width=2)
    ))

    # Mean and median
    mean_val = np.mean(distances)
    median_val = np.median(distances)

    max_y = max(y_kde_scaled)
    # Vertical lines with same color
    fig.add_shape(type="line", x0=mean_val, x1=mean_val, y0=0, y1=max_y, opacity=1,
                  line=dict(color=color, dash="dash"))
    fig.add_shape(type="line", x0=median_val, x1=median_val, y0=0, y1=max_y, opacity=1,
                  line=dict(color=color, dash="dot"))
    return fig, max_y
