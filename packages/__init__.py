from flask import Flask
from lime.lime_tabular import LimeTabularExplainer
from sklearn.neighbors import NearestNeighbors
import dash
import pandas as pd
import pickle

# Init server
server = Flask(__name__)
server.config['DEBUG'] = False

# Récupération des données
df = pd.read_csv('packages/ressources/app_encoded.csv', sep=',', index_col=0, encoding='utf8')
df_no_transformation = pd.read_csv(
    'packages/ressources/app_no_encoded_featureengineering_missing_value.csv',
    sep=',', index_col=0, encoding='utf8')
df_no_transformation = df_no_transformation.drop(columns=['Test'])

# Récupération du modèle
clf_pipe = pickle.load(open('packages/ressources/banking_model.pkl', 'rb'))

# Préparation du dataset avec les features selectionnés
x_test_transformed = pd.DataFrame(
    clf_pipe[0].transform(df.drop(columns=['TARGET', 'Test', 'SK_ID_CURR'])),
    columns=df.drop(columns=['TARGET', 'Test', 'SK_ID_CURR']).columns,
    index=df.index)

# Calcul des 20 plus proches voisins
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x_test_transformed)

# Interprétabilité du modèle
lime1 = LimeTabularExplainer(
    x_test_transformed,
    feature_names=x_test_transformed.columns,
    class_names=['Solvent', 'Insolvent'],
    discretize_continuous=False)

# Features
num_columns = df_no_transformation.select_dtypes(include=['float64']).columns

# Init dashboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,
                server=server,
                routes_pathname_prefix='/dash/',
                external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True

from packages import dashboard, functions, routes
