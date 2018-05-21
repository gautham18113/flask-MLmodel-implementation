"""Filename: server.py
"""

import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import dill as pickle
import warnings
warnings.filterwarnings("ignore")


appl = Flask(__name__)


@appl.route('/predict', methods=['POST'])
def apicall_post():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

        #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        #Getting the Loan_IDs separated out
        loan_ids = test['Loan_ID']

    except Exception as e:
        raise e

    clf = 'model_v1.pk'

    if test.empty:
        return(bad_request())
    else:
        #Load the saved model
        print("Loading the model...")
        loaded_model = None
        with open('%s\\model\\%s'%(os.getcwd(),clf),'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)),columns=['loan_id', 'prediction'])

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)


@appl.route('/predict', methods=['GET'])
def apicall_get():
    import json
    import requests
    from json2html import json2html
    import ast
    """Setting the headers to send and accept json responses
    """
    header = {'Content-Type': 'application/json', 'Accept': 'application/json'}

    """Reading test batch
    """
    df = pd.read_csv('%s\\data\\%s' % (os.getcwd(), "test.csv"), encoding="utf-8-sig")
    df.dropna(inplace=True)
    df = df[:]

    """Converting Pandas Dataframe to json
    """
    data = df.to_json(orient='records')
    """POST <url>/predict
    """
    resp = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data), headers=header)
    df = pd.DataFrame(ast.literal_eval(resp.json()['predictions']))
    return df.to_html()


if __name__=="__main__":
    appl.run(debug=True)

