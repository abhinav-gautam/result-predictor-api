from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import traceback


app = Flask(__name__)
CORS(app)

@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    if dtc_model1:
        try:
            _json = request.json
            query = pd.get_dummies(pd.DataFrame(_json))
            query = query.reindex(columns=model1_columns, fill_value=0)

            prediction = list(dtc_model1.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    if dtc_model2:
        try:
            _json = request.json
            query = pd.get_dummies(pd.DataFrame(_json))
            query = query.reindex(columns=model2_columns, fill_value=0)

            prediction = list(dtc_model2.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/predict_model3', methods=['POST'])
def predict_model3():
    if dtc_model3:
        try:
            _json = request.json
            query = pd.get_dummies(pd.DataFrame(_json))
            query = query.reindex(columns=model3_columns, fill_value=0)

            prediction = list(dtc_model3.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/predict_model4', methods=['POST'])
def predict_model4():
    if dtc_model4:
        try:
            _json = request.json
            query = pd.get_dummies(pd.DataFrame(_json))
            query = query.reindex(columns=model4_columns, fill_value=0)

            prediction = list(dtc_model4.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/predict_model5', methods=['POST'])
def predict_model5():
    if dtc_model5:
        try:
            _json = request.json
            query = pd.get_dummies(pd.DataFrame(_json))
            query = query.reindex(columns=model5_columns, fill_value=0)

            prediction = list(dtc_model5.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
        
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    dtc_model1 = joblib.load("model1.pkl") # Load "model.pkl"
    dtc_model2 = joblib.load("model2.pkl") # Load "model.pkl"
    dtc_model3 = joblib.load("model3.pkl") # Load "model.pkl"
    dtc_model4 = joblib.load("model4.pkl") # Load "model.pkl"
    dtc_model5 = joblib.load("model5.pkl") # Load "model.pkl"
    print ('Models loaded')
    model1_columns = joblib.load("model1_columns.pkl") # Load "model_columns.pkl"
    model2_columns = joblib.load("model2_columns.pkl") # Load "model_columns.pkl"
    model3_columns = joblib.load("model3_columns.pkl") # Load "model_columns.pkl"
    model4_columns = joblib.load("model4_columns.pkl") # Load "model_columns.pkl"
    model5_columns = joblib.load("model5_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    app.run(port=port, debug=True)