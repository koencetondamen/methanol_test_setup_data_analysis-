from typing import Optional, Dict, Any

import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
import pandas as pd

from ..acquisition import AcquisitionManager
from ..experiment_log import ExperimentLogger
from .. import config


def create_app(
    acquisition: AcquisitionManager,
    experiment_logger: ExperimentLogger,
) -> Dash:
    app = Dash(__name__)
    app.title = "Methanol / N₂ Test Dashboard"

    sensor_options = [
        {"label": f["label"], "value": f["field"]}
        for f in config.SENSOR_FIELDS
    ]

    app.layout = html.Div(
        [
            html.H1("Methanol / N₂ Test Dashboard"),

            # Experiment control ------------------------------------------------
            html.Div(
                [
                    html.H3("Experiment control"),
                    html.Div(
                        [
                            dcc.Input(
                                id="exp-name",
                                type="text",
                                placeholder="Experiment name",
                                style={"marginRight": "0.5rem"},
                            ),
                            dcc.Input(
                                id="exp-operator",
                                type="text",
                                placeholder="Operator",
                                style={"marginRight": "0.5rem"},
                            ),
                            dcc.Input(
                                id="exp-notes",
                                type="text",
                                placeholder="Notes",
                                style={"width": "300px", "marginRight": "0.5rem"},
                            ),
                        ],
                        style={"marginBottom": "0.5rem"},
                    ),
                    html.Button("Start experiment", id="btn-start-exp", n_clicks=0),
                    html.Button(
                        "Stop experiment",
                        id="btn-stop-exp",
                        n_clicks=0,
                        style={"marginLeft": "0.5rem"},
                    ),
                    html.Div(id="exp-status", style={"marginTop": "0.5rem"}),
                ],
                style={"border": "1px solid #ccc", "padding": "1rem", "marginBottom": "1rem"},
            ),

            # Live 11 sensors as cards -----------------------------------------
            html.Div(
                [
                    html.H3("Live sensor values (11 sensors)"),
                    html.Div(id="live-cards", style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                        "gap": "0.75rem",
                    }),
                ],
                style={"marginBottom": "1rem"},
            ),

            # Time-series plot --------------------------------------------------
            html.Div(
                [
                    html.H3("History plot"),
                    html.Div(
                        [
                            html.Label("Select signal:"),
                            dcc.Dropdown(
                                id="history-field",
                                options=sensor_options,
                                value=config.SENSOR_FIELDS[0]["field"],
                                clearable=False,
                                style={"width": "350px"},
                            ),
                        ],
                        style={"marginBottom": "0.5rem"},
                    ),
                    dcc.Graph(id="history-graph"),
                ]
            ),

            dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "sans-serif"},
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    @app.callback(
        Output("live-cards", "children"),
        Input("update-interval", "n_intervals"),
    )
    def update_live_cards(_n: int):
        latest = acquisition.get_latest()
        if latest is None:
            return html.Div("No data yet...")

        cards = []
        for spec in config.SENSOR_FIELDS:
            field = spec["field"]
            label = spec["label"]
            unit = spec["unit"]
            value = latest.get(field, None)

            text_value = "—"
            if value is not None:
                try:
                    text_value = f"{float(value):.3f} {unit}"
                except (TypeError, ValueError):
                    text_value = f"{value}"

            cards.append(
                html.Div(
                    [
                        html.Div(label, style={"fontSize": "0.9rem", "color": "#555"}),
                        html.Div(
                            text_value,
                            style={"fontSize": "1.4rem", "fontWeight": "bold"},
                        ),
                    ],
                    style={
                        "border": "1px solid #ddd",
                        "borderRadius": "8px",
                        "padding": "0.75rem",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                    },
                )
            )

        return cards

    @app.callback(
        Output("history-graph", "figure"),
        [Input("update-interval", "n_intervals"), Input("history-field", "value")],
    )
    def update_history_graph(_n: int, field: str):
        history = acquisition.get_history()
        if not history:
            return go.Figure(
                data=[],
                layout=go.Layout(
                    title="No data yet",
                    xaxis={"title": "Time"},
                    yaxis={"title": field},
                ),
            )

        df = pd.DataFrame(history)
        if field not in df.columns:
            return go.Figure(
                data=[],
                layout=go.Layout(
                    title=f"Field '{field}' not available yet",
                    xaxis={"title": "Time"},
                    yaxis={"title": field},
                ),
            )

        x = pd.to_datetime(df["timestamp_utc"])
        y = df[field]

        fig = go.Figure(
            data=[go.Scatter(x=x, y=y, mode="lines+markers", name=field)],
            layout=go.Layout(
                title=field,
                xaxis={"title": "Time"},
                yaxis={"title": field},
                margin={"l": 40, "r": 20, "t": 40, "b": 40},
            ),
        )
        return fig

    @app.callback(
        Output("exp-status", "children"),
        [
            Input("btn-start-exp", "n_clicks"),
            Input("btn-stop-exp", "n_clicks"),
        ],
        [
            State("exp-name", "value"),
            State("exp-operator", "value"),
            State("exp-notes", "value"),
        ],
        prevent_initial_call=True,
    )
    def handle_experiment_buttons(
        n_start: int,
        n_stop: int,
        name: Optional[str],
        operator: Optional[str],
        notes: Optional[str],
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "btn-start-exp":
            meta: Dict[str, Any] = {
                "name": name or "experiment",
                "operator": operator or "",
                "notes": notes or "",
            }
            experiment_logger.start_experiment(meta)
            return f"Experiment started: {meta['name']}"

        if button_id == "btn-stop-exp":
            experiment_logger.stop_experiment()
            return "Experiment stopped."

        raise dash.exceptions.PreventUpdate

    return app
