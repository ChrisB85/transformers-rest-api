from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Create a pipeline for automatic speech recognition using the specific model
asr_pipeline = pipeline(task="automatic-speech-recognition", model="bardsai/whisper-medium-pl", device=0 if torch.cuda.is_available() else -1)

@app.route('/convert_audio_to_text', methods=['POST'])
def convert_audio_to_text():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio']

        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400

        # Perform automatic speech recognition on the audio file
        text = asr_pipeline(audio_file.read())

        return jsonify(text), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
