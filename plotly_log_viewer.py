import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os

# --- Chargement du dernier fichier de log ---
log_dir = "logs"
log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("snake_dqn_log_20250326_113131_good_one.csv")])
latest_log = os.path.join(log_dir, log_files[-1])
df = pd.read_csv(latest_log)

# Colonnes numériques disponibles
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# --- Interface Dash ---
app = dash.Dash(__name__)
app.title = "Snake DQN Log Viewer"

app.layout = html.Div([
    html.H1("📊 Analyse des logs Snake DQN"),

    html.Label("Colonnes à afficher :"),
    dcc.Checklist(
        id='columns-selector',
        options=[{'label': col, 'value': col} for col in numeric_columns],
        value=['reward'],
        inline=True
    ),

    html.Br(),
    html.Label("Taille de la moyenne glissante :"),
    dcc.Slider(
        id='rolling-window',
        min=1,
        max=100,
        step=1,
        value=1,
        marks={1: '1', 10: '10', 25: '25', 50: '50', 100: '100'}
    ),

    dcc.Graph(id='log-graph')
])


@app.callback(
    Output('log-graph', 'figure'),
    [Input('columns-selector', 'value'),
     Input('rolling-window', 'value')]
)
def update_graph(selected_columns, window_size):
    fig = go.Figure()
    for col in selected_columns:
        if window_size > 1:
            smoothed = df[col].rolling(window=window_size).mean()
            fig.add_trace(go.Scatter(x=df['episode'], y=smoothed, mode='lines', name=f"{col} (MA {window_size})"))
        else:
            fig.add_trace(go.Scatter(x=df['episode'], y=df[col], mode='lines', name=col))

    fig.update_layout(
        title="Courbes d'entraînement",
        xaxis_title="Épisode",
        yaxis_title="Valeur",
        hovermode='x unified',
        template='plotly_dark'
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)
