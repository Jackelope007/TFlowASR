from flask import Flask, request
import numpy as np
import tensorflow as tf
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.configs.config import Config

app = Flask(__name__)

# Load the configuration files
speech_config = Config("./config.yml")
text_config = Config("./config.yml")

# Initialize the featurizers
speech_featurizer = TFSpeechFeaturizer(speech_config)
text_featurizer = CharFeaturizer(text_config)

# Initialize the model
model = Conformer(**speech_config["model_config"], vocabulary_size=text_featurizer.num_classes)
model.make(speech_featurizer.shape)
model.load_weights("/app/conformer.h5")

# Transcribe the audio file
def transcribe(audio):
    features = speech_featurizer.extract(audio)
    input_length = np.array([features.shape[0]])
    features = tf.expand_dims(features, axis=0)
    predicted = model.recognize(features, input_length)
    return text_featurizer.iextract(predicted.numpy())

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        audio_file = request.files['file']

        # Save the audio file
        audio_file.save(audio_file.filename)

        # Transcribe the audio file
        with open(audio_file.filename, 'rb') as fin:
            audio = np.frombuffer(fin.read(), np.int16)
        text = transcribe(audio)

        return text

    return '''
    <!doctype html>
    <title>Upload an audio file</title>
    <h1>Upload an audio file and get its transcription</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
