import os

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request
from keras.models import load_model

app = Flask(__name__)

SIZE = 224
SIZE2 = 64
class_labels = ['Class 0 (akiec)', 'Class 1 (bcc)', 'Class 2 (bkl)', 'Class 3 (df)', 'Class 4 (mel)',
                    'Class 5 (nv)', 'Class 6 (vasc)']


@app.route('/', methods=["GET"])
def hello_world():
    # model = load_model(
    #     os.path.join('Densenetmodel50epochs1500resample224size.keras'))
    #
    # image = Image.open('df.jpg')
    # image = tf.image.resize(image, (SIZE, SIZE))
    # image = np.expand_dims(image / 255, axis=0)
    #
    # prediction = model.predict(image)
    #
    # predicted_label = 'Predicted class is: ' + str(
    #     class_labels[prediction.argmax()]) + '. With a probability of ' + str(prediction[0, prediction.argmax()]) + '.'

    #dictToReturn = {'': str(predicted_label)}
    dictToReturn = {'': 'hello world'}

    #return jsonify(dictToReturn)
    return dictToReturn

@app.route('/post', methods=["POST"])
def testpost():
     input_json = request.get_json(force=True)
     dictToReturn = {'text':input_json['text']}
     return jsonify(dictToReturn)

@app.route('/prediction', methods=["POST"])
def prediction():
    #model_path = '"C:\\Users\jesse\OneDrive\Desktop\Year 4\Project\models\Densenetmodel50epochs1500resample224size.keras"'
    model = load_model(
        os.path.join('Densenetmodel50epochs1500resample224size.keras'))

    #file = request.files['file']
    img = request.files['file']
    image = Image.open(img)
    image = tf.image.resize(image, (SIZE, SIZE))
    image = np.expand_dims(image / 255, axis=0)

    prediction = model.predict(image)

    predicted_label = 'Predicted class is: ' + str(class_labels[prediction.argmax()]) + '. With a probability of ' + str(prediction[0, prediction.argmax()]) + '.'

    dictToReturn = {'': str(predicted_label)}

    return jsonify(dictToReturn)


@app.route('/cnnsvm', methods=["POST"])
def SVM_CNN_prediction():
    model = load_model(
        os.path.join('Densenetmodel50epochs1500resample224size.keras'))

    svm = joblib.load(os.path.join('svm_model.joblib'))

    img = request.files['file']
    image = Image.open(img)
    image = tf.image.resize(image, (SIZE, SIZE))
    image = np.expand_dims(image / 255, axis=0)

    features = model.predict(image)

    prediction = svm.predict(features)

    predicted_label = class_labels[prediction[0]]

    dictToReturn = {'': str(predicted_label)}

    return jsonify(dictToReturn)


@app.route('/pretrained', methods=["POST"])
def pretrainedCnn():
    model = load_model(os.path.join('SequentialRelu50epochs1500resample64size.keras'))

    img = request.files['file']
    image = Image.open(img)
    image = tf.image.resize(image, (SIZE2, SIZE2))
    image = np.expand_dims(image / 255, axis=0)

    prediction = model.predict(image)

    predicted_label = 'Predicted class is: ' + str(class_labels[prediction.argmax()]) + '. With a probability of ' + str(prediction[0, prediction.argmax()]) + '.'

    dictToReturn = {'': str(predicted_label)}

    return jsonify(dictToReturn)


if __name__ == '__main__':
    app.run(debug=True)