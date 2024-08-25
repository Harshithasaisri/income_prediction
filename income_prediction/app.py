from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(app.root_path, 'incomeprediction_savedmodel.sav')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Mapping for categorical inputs
workclass_mapping = {'State-gov': 0, 'Private': 1}
education_mapping = {'Bachelors': 0, 'HS-grad': 1}
occupation_mapping = {'Adm-clerical': 0, 'Exec-managerial': 1}
sex_mapping = {'Male': 0, 'Female': 1}
country_mapping = {'United-States': 0, 'Cuba': 1, 'India': 2}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        if request.method == 'POST':
            # Retrieve the input data
            age = int(request.form['age'])
            workclass = request.form['workclass']
            education = request.form['education']
            occupation = request.form['occupation']
            sex = request.form['sex']
            capital_gain = int(request.form['capital-gain'])
            capital_loss = int(request.form['capital-loss'])
            hours_per_week = int(request.form['hours-per-week'])
            native_country = request.form['native-country']

            # Convert the categorical data to numerical values
            workclass = workclass_mapping.get(workclass, 0)
            education = education_mapping.get(education, 0)
            occupation = occupation_mapping.get(occupation, 0)
            sex = sex_mapping.get(sex, 0)
            native_country = country_mapping.get(native_country, 0)

            # Prepare the data for prediction
            input_data = np.array([[age, workclass, education, occupation, sex, capital_gain, capital_loss, hours_per_week, native_country]])
            prediction = model.predict(input_data)

            # Convert numerical prediction to categorical output
            output = '>50K' if prediction[0] == 1 else '<=50K'

            return render_template('result.html', output=output)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
