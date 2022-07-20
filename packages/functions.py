import numpy as np
import pandas as pd
import plotly.graph_objs as go
from operator import itemgetter
from packages import df_no_transformation, clf_pipe, lime1, nbrs, x_test_transformed

informations_to_display = {
    'NAME_INCOME_TYPE' : 'Income type',
    'AMT_INCOME_TOTAL' : 'Amount of income',
    'AMT_CREDIT' : 'Amount of credit',
    'AMT_GOODS_PRICE' : 'Amount of goods price',
    'AMT_ANNUITY' : 'Amount of annuity',
    'INCOME_TO_CREDIT_RATIO' : 'Ratio Income / Credit',
    'CREDIT_TO_ANNUITY_RATIO' : 'Ratio Credit / Annuity',
    'INCOME_TO_FAMILYSIZE_RATIO' : 'Ratio Income / Family size',
    'NAME_FAMILY_STATUS' : 'Family status',
    'EXT_SOURCE_2' : 'EXT_SOURCE_2'
}

def isFloat(var):
    try:
        float(var)
        return True
    except ValueError:
        return False

def find_loan(id_loan):
    if id_loan in df_no_transformation['SK_ID_CURR']:
        return True
    return False

def get_info_loan(id_loan):
    result = dict()
    for col_name in list(informations_to_display):
        if isFloat(df_no_transformation[col_name][id_loan]):
            result[informations_to_display[col_name]] = round(float(df_no_transformation[col_name][id_loan]),6)
        else:
            result[informations_to_display[col_name]] = df_no_transformation[col_name][id_loan]
    return result

def get_solvency(id_loan):
    return clf_pipe[1].predict_proba(np.array(x_test_transformed.loc[id_loan]).reshape(1, -1)).flatten()

def get_feature_importance(id_loan):
    exp = lime1.explain_instance(x_test_transformed.loc[id_loan], clf_pipe[1].predict_proba, num_samples=100)
    indices, values = [], []
    for ind, val in sorted(exp.as_list(), key=itemgetter(1)):
        indices.append(ind)
        values.append(val)
    data = pd.DataFrame(values, columns=["values"], index=indices)
    del indices, values
    data["positive"] = data["values"]>0
    return data

def get_similarity(id_loan):
    indices_similary_loans = nbrs.kneighbors(np.array(x_test_transformed.loc[id_loan]).reshape(1, -1))[1].flatten()
    return df_no_transformation.iloc[indices_similary_loans].reset_index().drop(columns="index")

# [WIP]
def get_global_feature_importances(n_top_features=20):
    indices, values = [], []
    for ind, val in sorted(zip(
        clf_pipe[1].feature_importances_,
        x_test_transformed.columns),
        reverse=True)[0:  n_top_features] :
        indices.append(ind)
        values.append(val)
    data = pd.DataFrame(values, columns=["values"], index=indices)
    del indices, values
    return {
        'data': [go.Bar(
                    x=data.index,
                    y=data["values"],
                    orientation='h',
        )],

        'layout': go.Layout(
                            margin={'l': 300, 'b': 50, 't': 30, 'r': 30},
                            height=700,
                            width=1200
                            )
    }
