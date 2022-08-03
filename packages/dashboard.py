import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
from packages.functions import get_feature_importance, get_info_loan, get_similarity, get_solvency
from . import app, df_no_transformation, num_columns, solvency_threshold, x_test_transformed

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app.layout = html.Div([

    dcc.Tabs([
        # Premier onglet: Solvency
        dcc.Tab(id='Solvency', label='Solvency', children=[
            html.Div([
                # Permet de séléctionner dans une liste déroulante le numéro de l'emprunt
                html.Div([
                    html.Div([
                        html.H3("Id loan"),
                        dcc.Dropdown(
                            id='id-loan',
                            options=[{'label': i, 'value': i} for i in x_test_transformed.index],
                            value=x_test_transformed.index[0],
                            clearable=False,
                            style={'width': '100%'}
                        ),
                    ]),
                    html.Div(id='info-loan')
                ], style={'width': '24%', 'float': 'left'}),
                # Affiche la probabilité de solvabilité d'un client
                # sous forme de pie plot
                html.Div([
                    html.H3("Probability of Solvency"),
                    html.H5("Solvency threshold : "+ str(solvency_threshold)),
                    dcc.Graph(
                        id='solvency',
                        figure={},
                    ),
                ], style={'width': '29%', 'float': 'center'}),
                # Affiche pour l'emprunt séléctionné
                # l'importance des features qui ont eu le plus d'impacte
                # sur la solvabilité d'un client ou non
                html.Div([
                    html.H3("Feature Importances"),
                    dcc.Graph(
                        id='graph',
                        figure={},
                    ),
                ], style={'width': '42%', 'float': 'right'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between'}),
            # Affiche un tableau contenant les informations relatives
            # à l'emprunt séléctionné ainsi que les emprunts similaires
            html.Div([
                html.H3("Similarity"),
                dash_table.DataTable(
                    id='table',
                    columns=[
                        {"name": i, "id": i} for i in df_no_transformation.reset_index().drop(columns='index').columns
                    ],
                    fixed_rows={'headers': True, 'data': 0 },
                    style_cell={'width': '200px'},
                    style_table={'minWidth': '80%'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                    virtualization=True,
                ),
            ]),
        ]),
        # Deuxième onglet [WIP]
        dcc.Tab(label='Data exploration', children=[
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='xaxis-column',
                        options=[{'label': i, 'value': i} for i in num_columns],
                        value='AMT_CREDIT'
                    ),
                    dcc.RadioItems(
                        id='xaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],
                style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Dropdown(
                        id='yaxis-column',
                        options=[{'label': i, 'value': i} for i in num_columns],
                        value='AMT_ANNUITY'
                    ),
                    dcc.RadioItems(
                        id='yaxis-type',
                        options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                        value='Linear',
                        labelStyle={'display': 'inline-block'}
                    )
                ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            dcc.Graph(id='indicator-graphic'),
        ]),
    ]),
])

# Met à jour les informations du client
@app.callback(
    Output('info-loan', 'children'),
    [Input('id-loan', 'value')])
def update_info_loan(id_loan):
    info_loan = get_info_loan(id_loan)
    return html.Ul([
        html.Li([''+str(i)+' : '+str(info_loan[i])]) for i in info_loan
    ])

# Met à jour le pieplot de la solvabilité du client
@app.callback(
    Output('solvency', 'figure'),
    [Input('id-loan', 'value')])
def update_pieplot_solvency(id_loan):
    values = get_solvency(id_loan)
    if values[0] > solvency_threshold:
        solvency_result = ['SOLVENT', 'Insolvent']
    else:
        solvency_result = ['Solvent', 'INSOLVENT']
    return {
        'data': [go.Pie(labels=solvency_result,
                        values=values,
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=.3)],
        'layout': go.Layout(margin=dict(b=100))
    }

# Met à jour le graphique de l'importance des features pour
# le client dont l'id est séléctionné
@app.callback(
    Output('graph', 'figure'),
    [Input('id-loan', 'value')])
def update_barplot_feat_imp(id_loan) :
    data = get_feature_importance(id_loan)
    return {
        'data': [go.Bar(x=data["values"],
                        y=data.index,
                        orientation='h',
                        marker_color=list(data.positive.map({True: '#2ecc71', False: '#e74c3c'}).values))],
        'layout': go.Layout(margin=dict(l=300, r=0, t=30, b=100))
    }

# Met à jour le tableau de prêts similaires
# au prêt dont l'id est choisi
@app.callback(
    Output('table', 'data'),
    [Input('id-loan', 'value')])
def update_table_similarity(id_loan):
    df_similarity = get_similarity(id_loan)
    return df_similarity.to_dict('records')

# Met à jour le graphe pour l'exploration de données
@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
    Input('yaxis-column', 'value'),
    Input('xaxis-type', 'value'),
    Input('yaxis-type', 'value')])
def update_graph_data_exploration(xaxis_column, yaxis_column, xaxis_type, yaxis_type):
    traces = []
    for i, target in enumerate(df_no_transformation.TARGET.unique()):
        filtered_df = df_no_transformation[df_no_transformation['TARGET'] == target].reset_index()
        traces.append(dict(
            x=filtered_df[xaxis_column],
            y=filtered_df[yaxis_column],
            text=filtered_df['SK_ID_CURR'],
            mode='markers',
            opacity=0.7,
            marker={
                'color':list(filtered_df['TARGET'].map({1.0: '#e74c3c', 0.0: '#2ecc71'}).values),
                'size': 5,
                'line': {'width': 0.10, 'color': 'white'}
            },
        ))
    return {
        'data': traces,
        'layout': dict(
            xaxis={
                'title': xaxis_column,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }
