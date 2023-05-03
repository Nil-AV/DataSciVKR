from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])

@app.route('/composite/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('main.html')
    if request.method == 'POST':
        param_list = ('var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'var10', 'var11', 'var12')
        params = []
        for i in param_list:
            param = request.form.get(i)
            params.append(param)
        params = [float(i.replace(',', '.')) for i in params]
        
        scaler_X = joblib.load('./models/X.joblib')
        params = scaler_X.inverse_transform([params])
        model = tf.keras.models.load_model('models/model.h5')
        pred = model.predict([params])
        scaler_y = joblib.load('./models/y.joblib')
        pred = scaler_y.inverse_transform(pred)

        message = f'Спрогнозированное cоотношение матрица-наполнитель для введенных параметров: {pred}'
    return render_template('main.html', message=message)


if __name__ == '__main__':
    app.run()
