import joblib
import numpy as np
import tensorflow as tf
from app.core.config import settings
from app.models.prediction import PredictionResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    # Muat model
    loaded_model = tf.keras.models.load_model(settings.MODEL_PATH)
    print(f"Model loaded successfully with input shape: {loaded_model.input_shape}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

try:
    # Muat tokenizer
    tokenizer = joblib.load(settings.VECTORIZER_PATH)
    print(f"Tokenizer loaded successfully: {type(tokenizer)}")
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer: {str(e)}")

# Panjang input yang diharapkan oleh model
expected_input_length = loaded_model.input_shape[1]

def predict_text(input_text: str) -> PredictionResponse:
    # Ubah teks menjadi sequence angka
    tokenized_text = tokenizer.texts_to_sequences([input_text])

    # Lakukan padding sequence agar sesuai dengan panjang input model
    padded_text = pad_sequences(tokenized_text, maxlen=expected_input_length)

    # Prediksi menggunakan model
    prediction = loaded_model.predict(padded_text)
    label = "jamid" if prediction[0][0] >= 0.5 else "musytaq"

    return PredictionResponse(text=input_text, prediction=label)
