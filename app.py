from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['POST', 'GET'])
def index():
     return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final = [np.array(features)]
        Predict = model.predict(final)

        output = round(Predict[0], 4)
        if(float(output)<100):
            return (render_template('index.html', prediction_text='Predicted Percentile is {}'.format(output)))
        else:
            return (render_template('index.html', prediction_text='Predicted Percentile is {}'.format('100')))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)