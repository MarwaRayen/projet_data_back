import datetime
import yfinance as yf
import pandas as pd
import matplotlib as mt
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import json
from .finance import int_to_date,date_to_int


# Nouvelle palette violet/rose/dark
colors = [
    "rgba(168,85,247,1)",   # Purple-500
    "rgba(236,72,153,1)",   # Pink-500
    "rgba(139,92,246,1)",   # Violet-500
    "rgba(217,70,239,1)",   # Fuchsia-500
    "rgba(124,58,237,1)",   # Purple-600
    "rgba(219,39,119,1)",   # Pink-600
    "rgba(147,51,234,1)",   # Purple-600
    "rgba(192,132,252,1)",  # Purple-400
    "rgba(244,114,182,1)",  # Pink-400
    "rgba(167,139,250,1)",  # Violet-400
]

def get_plotly_histogram(title: str, xaxis_title: str, yaxis_title: str):
    fig = go.Figure()
    fig.update_layout(
        title={
            "text": title,
            "font": {"color": "white", "size": 20, "family": "Arial, sans-serif"},
        },
        xaxis={
            "title": {"text": xaxis_title, "font": {"color": "rgba(192,132,252,1)", "size": 14}},
            "tickfont": {"color": "rgba(192,132,252,1)"},
            "gridcolor": "rgba(124,58,237,0.2)",
            "showgrid": True,
        },
        yaxis={
            "title": {"text": yaxis_title, "font": {"color": "rgba(192,132,252,1)", "size": 14}},
            "tickfont": {"color": "rgba(192,132,252,1)"},
            "gridcolor": "rgba(124,58,237,0.2)",
            "showgrid": True,
        },
        plot_bgcolor="rgba(15,15,30,1)",  # #0f0f1e
        paper_bgcolor="rgba(22,33,62,1)",  # #16213e
        font={"color": "rgba(192,132,252,1)"},
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            bgcolor="rgba(26,26,46,0.8)",
            bordercolor="rgba(124,58,237,0.5)",
            borderwidth=1,
            font=dict(color="white")
        ),
        hovermode='x unified',
    )
    return fig

def get_plot_histogram(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_histogram("Distribution de l'investissement par actif", "Investissement (USD)", "Fréquence")
    
    for idx, (key, df_actif) in enumerate(df_multi_actifs.T.groupby(level=0)):
        color = colors[idx % len(colors)]
        figure.add_trace(go.Histogram(
            x=df_actif.droplevel(0).T['investissement_cumule'], 
            name=key,
            nbinsx=20,
            marker=dict(
                color=color,
                line=dict(color="rgba(255,255,255,0.3)", width=1)
            ),
            opacity=0.8
        ))
    
    return figure


def get_plotly_figure(title: str, xaxis_title: str, yaxis_title: str):
    fig = go.Figure()
    fig.update_layout(
        title={
            "text": title,
            "font": {"color": "white", "size": 20, "family": "Arial, sans-serif"},
        },
        xaxis={
            "title": {"text": xaxis_title, "font": {"color": "rgba(192,132,252,1)", "size": 14}},
            "tickfont": {"color": "rgba(192,132,252,1)"},
            "gridcolor": "rgba(124,58,237,0.2)",
            "showgrid": True,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": yaxis_title, "font": {"color": "rgba(192,132,252,1)", "size": 14}},
            "tickfont": {"color": "rgba(192,132,252,1)"},
            "gridcolor": "rgba(124,58,237,0.2)",
            "showgrid": True,
            "zeroline": False,
        },
        plot_bgcolor="rgba(15,15,30,1)",  # #0f0f1e
        paper_bgcolor="rgba(22,33,62,1)",  # #16213e
        font={"color": "rgba(192,132,252,1)"},
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            bgcolor="rgba(26,26,46,0.8)",
            bordercolor="rgba(124,58,237,0.5)",
            borderwidth=1,
            font=dict(color="white", size=12),
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor="rgba(26,26,46,0.95)",
            font_size=12,
            font_family="Arial",
            bordercolor="rgba(168,85,247,0.8)"
        )
    )
    return fig


def add_trace_plotly(fig: go.Figure, trace_data: pd.Series, label: str, mode: str, legendgroup=None, visible=True, color=None):
    trace_config = {
        'x': trace_data.index,
        'y': trace_data,
        'mode': mode,
        'name': label,
        'showlegend': True,
        'legendgroup': legendgroup,
        'visible': visible,
        'line': dict(width=2.5),
        'hovertemplate': f'<b>{label}</b><br>%{{y:.2f}}<extra></extra>'
    }
    
    if color:
        trace_config['line']['color'] = color
    
    fig.add_trace(go.Scatter(**trace_config))
    return fig

def convert_plotly_to_json(figure):
    json_plot = plotly.io.to_json(figure)
    return json_plot

def get_plot_adj_close(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_figure(
        "Evolution du cours de clôture ajusté",
        "Temps (mois et année)",
        "Cours de clôture ajusté (USD)"
    )
    
    idx = 0
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        if key != "TOTAL":
            color = colors[idx % len(colors)]
            figure = add_trace_plotly(
                figure,
                df_actif.droplevel(0).T['adj_close'],
                key,
                "lines",
                color=color
            )
            idx += 1

    return figure

def get_plot_rendement(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_figure(
        "Evolution du rendement par actif",
        "Temps (mois / trimestre / année)",
        "Rendement (USD)"
    )
    
    idx = 0
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        color = colors[idx % len(colors)]
        figure = add_trace_plotly(
            figure,
            df_actif.droplevel(0).T['rendement'],
            key,
            "lines",
            color=color
        )
        idx += 1
    
    return figure

def get_plot_investissement(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_figure(
        "Evolution de l'investissement cumulé par actif",
        "Temps (mois et année)",
        "Investissement cumulé (USD)"
    )
    
    idx = 0
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        color = colors[idx % len(colors)]
        figure = add_trace_plotly(
            figure,
            df_actif.droplevel(0).T['investissement_cumule'],
            key,
            "lines",
            color=color
        )
        idx += 1
    
    return figure

def get_table_stats(df_stats: pd.DataFrame):
    # Format the data
    df_stats = df_stats.apply(
        lambda x: x.apply(lambda x: f"{x*100:.2f} %") if x.name != "ratio_sharpe" else x.apply(lambda x: f"{x:.3f}"),
        axis=1
    )

    headers = ['Mesure'] + list(df_stats.columns)
    cell_values = [df_stats.index.tolist()] + [df_stats[col].tolist() for col in df_stats.columns]

    # Create the table figure with purple/pink gradient theme
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="rgba(124,58,237,1)",  # Purple gradient
                    font=dict(color="white", size=14, family="Arial, sans-serif"), 
                    align="center",
                    height=50,
                    line=dict(color="rgba(168,85,247,0.5)", width=1)
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=[
                        ["rgba(26,26,46,0.8)", "rgba(22,33,62,0.8)"] * len(df_stats)  # Alternating dark rows
                    ],
                    font=dict(color="white", size=12),  
                    align="center",
                    line=dict(color="rgba(124,58,237,0.3)", width=1),  
                    height=40
                )
            )
        ]
    )

    # Update the layout
    fig.update_layout(
        title={
            "text": "Mesures financières associées au portefeuille",
            "font": {"color": "white", "size": 20, "family": "Arial, sans-serif"}
        },
        paper_bgcolor="rgba(22,33,62,1)",  # #16213e
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def get_plot_prediction_rendement(df_multi_actifs: pd.DataFrame):
    figure = get_plotly_figure(
        "Prédiction du rendement par régression linéaire, ± 1 et 2 écart-types",
        "Temps (mois et année)",
        "Rendement (USD)"
    )

    idx = 0
    for key, df_actif in df_multi_actifs.T.groupby(level=0):
        rendement_series = df_actif.droplevel(0).T['rendement']
        
        # Couleur principale pour l'actif
        main_color = colors[idx % len(colors)]
        
        # Ligne de données réelles
        figure = add_trace_plotly(
            figure,
            rendement_series,
            key,
            "lines",
            legendgroup=key,
            visible=(True if key == "TOTAL" else "legendonly"),
            color=main_color
        )

        linear_model = LinearRegression()
        X = pd.Series(pd.to_datetime(rendement_series.index)).apply(lambda date: date_to_int(date)).to_numpy().reshape(-1, 1)
        y = rendement_series
        linear_model.fit(X, y)
        y_pred = linear_model.predict(X)
        ecart_type = np.std(rendement_series - y_pred)

        time_axis = int_to_date(X.reshape(-1))

        # Ligne de prédiction
        figure.add_trace(
            go.Scatter(
                x=time_axis,
                y=y_pred,
                mode='lines',
                line=dict(dash='solid', width=2, color=main_color),
                name='Prédiction linéaire',
                legendgroup=key,
                showlegend=False,
                visible=(True if key == "TOTAL" else "legendonly"),
                opacity=0.8,
                hovertemplate='<b>Prédiction</b><br>%{y:.2f}<extra></extra>'
            )
        )

        # Bandes d'écart-type avec dégradé violet/rose
        ecart_colors = [
            "rgba(168,85,247,0.3)",   # Purple transparent pour -2σ
            "rgba(236,72,153,0.3)",   # Pink transparent pour -1σ
            "rgba(236,72,153,0.3)",   # Pink transparent pour +1σ
            "rgba(168,85,247,0.3)",   # Purple transparent pour +2σ
        ]
        
        color_idx = 0
        for nb_ecart_type in [2, 1]:  # Inverser l'ordre pour dessiner d'abord les grandes bandes
            constante_ecart_type = nb_ecart_type * ecart_type
            for ecart_val, neg_symbole in zip([-constante_ecart_type, constante_ecart_type], ["-", "+"]):
                figure.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=y_pred + ecart_val,
                        mode='lines',
                        line=dict(dash='dash', width=1.5, color=ecart_colors[color_idx]),
                        name=f"{neg_symbole}{nb_ecart_type}σ",
                        legendgroup=key,
                        showlegend=False,
                        visible=(True if key == "TOTAL" else "legendonly"),
                        opacity=0.6,
                        hovertemplate=f'<b>{neg_symbole}{nb_ecart_type}σ</b><br>%{{y:.2f}}<extra></extra>'
                    )
                )
                color_idx += 1

        idx += 1

    return figure
