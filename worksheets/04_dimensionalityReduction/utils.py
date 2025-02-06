
import plotly.graph_objs as go


def getHovertext(times, neurons_ids, z,
                 channels_for_neurons, regions_for_channels):
    hovertext = []
    for yi, yy in enumerate(neurons_ids):
        channel_for_neuron = channels_for_neurons[yi]
        region_for_neuron = regions_for_channels[channel_for_neuron]
        hovertext.append([])
        for xi, xx in enumerate(times):
            hovertext[-1].append(f"time: {xx}<br />neuron_id: {yy}<br />"
                                 f"z-score: {z[yi][xi]}<br />"
                                 f"loc: {region_for_neuron}")
    return hovertext


def getHeatmap(xs, ys, zs, hovertext, zmin, zmax, x_label, y_label, title="",
               colorbar_title=""):
    xzero = zmin/(zmin - zmax)
    colorscale = [[0.0, 'rgba(0, 0, 255, 0.85)'],
                  [xzero, 'rgba(255, 255, 255, 0.85)'],
                  [1.0, 'rgba(255, 0, 0, 0.85)']]
    fig = go.Figure()
    trace = go.Heatmap(x=xs, y=ys, z=zs,
                       zmin=zmin, zmax=zmax,
                       colorscale=colorscale, hoverinfo="text", text=hovertext,
                       colorbar={"title": colorbar_title})
    fig.add_trace(trace)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title)
    return fig

