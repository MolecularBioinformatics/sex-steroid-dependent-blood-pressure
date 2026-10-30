import numpy as np
import plotly.graph_objects as go

def softplus(x):
    return np.log(1 + np.exp(x))

if __name__ == "__main__":
    x = np.linspace(-5, 5, 99)
    y_relu = np.array([max(0, i) for i in x])
    y_softplus = softplus(x)

    fig = go.Figure()
    fig.add_trace(
            go.Scattergl(
                x=x,
                y=y_softplus,
                mode="lines",
                name="softplus(x)",                     
                line=dict(color="blue", width=3)
            )
        )
    fig.add_trace(
            go.Scattergl(
                x=x,
                y=y_relu,
                mode="lines",
                name="max(0, x)",                     
                line=dict(color="red", width=3)
            )
        )

    fig.update_yaxes(
        tickfont=dict(size=27),
        title_font=dict(size=33),
        title_text="Y"
    )

    fig.update_xaxes(
        tickfont=dict(size=27),
        title_font=dict(size=33),
        title_text="X",
        side="bottom"
    )
    
    fig.update_layout(
        title=dict(
            text=f"softplus(x) vs max(0, x)",
            font=dict(size=40, color="black"),
            x=0.5,  # center
            xanchor='center'
        ),
        legend=dict(
            font=dict(size=33),
            itemwidth=40,
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top"
        )
    )

    fig.show()
