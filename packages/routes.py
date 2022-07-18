from flask import render_template, request
from . import functions, server
import os
import plotly.graph_objs as go

if not os.path.exists('images'):
    os.mkdir('images')

@server.route('/', methods=['GET', 'POST'])
def index(id_loan=None):
    if request.method == 'POST':
        id_loan = int(request.form['id_loan'])
        info_loan = functions.get_info_loan(id_loan)

        # Solvency
        solvency = functions.get_solvency(id_loan)
        fig = go.Figure(
            data=[go.Pie(
                labels=['Solvent', 'Insolvent'],
                values=solvency,
                marker_colors=["#2ecc71", "#e74c3c"],
                hole=.3)],
            layout=go.Layout(margin=dict(b=100))
        )
        fig.write_image('images/pieplot_solvency.png')
        if solvency[0] > 0.90:
            solvency_result = str(round(solvency[0],4)*100)+'% - Solvent'
        else:
            solvency_result = str(round(solvency[0],4)*100)+'% - Insolvent'

        # Feature importance
        feat_imp = functions.get_feature_importance(id_loan)
        fig = go.Figure(
            data=[go.Bar(
                x=feat_imp["values"],
                y=feat_imp.index,
                orientation='h',
                marker_color=list(feat_imp.positive.map({True: '#e74c3c', False: '#2ecc71'}).values))],
            layout=go.Layout(margin=dict(l=300, r=0, t=30, b=100))
        )
        fig.write_html('images/feat_imp.png')

        #Similarity
        similarity = functions.get_similarity(id_loan).to_html()

        return render_template(
            'index.html',
            id_loan=id_loan,
            info_loan=info_loan,
            solvency=solvency_result,
            similarity=similarity
        )
    else:
        return render_template('index.html')
