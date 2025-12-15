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

            # Manual actions / events -----------------------------------------
            html.Div(
                [
                    html.H3("Actions / event log"),
                    html.Div(
                        [
                            dcc.Input(
                                id="action-text",
                                type="text",
                                placeholder="e.g., start pump / close valve 5",
                                style={"width": "420px", "marginRight": "0.5rem"},
                            ),
                            html.Button("Perform action", id="btn-action", n_clicks=0),
                        ],
                        style={"marginBottom": "0.5rem"},
                    ),
                    html.Div(id="action-status", style={"marginBottom": "0.5rem", "color": "#555"}),
                    html.Div(
                        id="event-log",
                        style={
                            "border": "1px solid #ddd",
                            "borderRadius": "8px",
                            "padding": "0.5rem",
                            "maxHeight": "220px",
                            "overflowY": "auto",
                            "backgroundColor": "#fafafa",
                        },
                    ),
                ],
                style={"border": "1px solid #ccc", "padding": "1rem", "marginBottom": "1rem"},
            ),

            # Live 11 sensors as cards -----------------------------------------
            html.Div(
                [
                    html.H3("Live sensor values"),
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
                    html.H3("History plots"),
                    html.Div(id="history-graphs"),
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
        [Output("action-status", "children"), Output("action-text", "value")],
        Input("btn-action", "n_clicks"),
        State("action-text", "value"),
        prevent_initial_call=True,
    )
    def handle_action_button(n_clicks: int, action_text: Optional[str]):
        action = (action_text or "").strip()
        if not action:
            return "Type an action first.", action_text

        if not experiment_logger.active:
            return "No active experiment — start an experiment before logging actions.", action_text

        experiment_logger.log_event(action)
        return f"Logged action: {action}", ""


    @app.callback(
        Output("event-log", "children"),
        Input("update-interval", "n_intervals"),
    )
    def update_event_log(_n: int):
        events = experiment_logger.get_recent_events(limit=50)

        if not events:
            return html.Div("No events yet.")

        # newest first
        rows = []
        for e in reversed(events):
            ts = e.get("timestamp_ams", "")
            hhmmss = ts[11:19] if len(ts) >= 19 else ts   # "YYYY-MM-DDTHH:MM:SS" -> "HH:MM:SS"

            rows.append(
                html.Tr(
                    [
                        html.Td(hhmmss, style={"whiteSpace": "nowrap", "paddingRight": "0.75rem"}),
                        html.Td(e.get("event", "")),
                    ]
                )
            )

        return html.Table(
            [
                html.Thead(html.Tr([html.Th("Time (UTC)"), html.Th("Action")])),
                html.Tbody(rows),
            ],
            style={"width": "100%", "fontSize": "0.95rem"},
        )


    @app.callback(
        Output("history-graphs", "children"),
        Input("update-interval", "n_intervals"),
    )
    def update_history_graphs(_n: int):
        history = acquisition.get_history()
        if not history:
            return [html.Div("No data yet...")]

        df = pd.DataFrame(history)

        # pick a timestamp column that exists
        time_col = "timestamp_ams" if "timestamp_ams" in df.columns else "timestamp_utc"
        if time_col not in df.columns:
            return [html.Div("No timestamp column in history yet.")]

        x = pd.to_datetime(df[time_col], errors="coerce")

        graphs = []
        for spec in config.SENSOR_FIELDS:
            field = spec["field"]
            label = spec["label"]
            unit = spec.get("unit", "")

            if field not in df.columns:
                continue

            y = df[field]

            # Only show plots for sensors that actually have values
            if y.dropna().empty:
                continue

            fig = go.Figure(
                data=[go.Scatter(x=x, y=y, mode="lines", name=label)],
                layout=go.Layout(
                    title=f"{label}{f' ({unit})' if unit else ''}",
                    xaxis={"title": "Time"},
                    yaxis={"title": unit or field},
                    margin={"l": 40, "r": 20, "t": 40, "b": 40},
                    height=280,
                ),
            )

            graphs.append(
                html.Div(
                    dcc.Graph(
                        figure=fig,
                        id={"type": "history-graph", "field": field},  # optional but nice
                    ),
                    style={"marginBottom": "0.75rem"},
                )
            )

        return graphs or [html.Div("No sensor values yet...")]

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
