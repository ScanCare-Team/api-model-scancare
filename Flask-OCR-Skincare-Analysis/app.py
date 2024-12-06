from flask import Flask, request, jsonify
import easyocr
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Inisialisasi EasyOCR reader
reader = easyocr.Reader(['en', 'id'])

# Muat model machine learning dan tokenizer
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Fungsi untuk memproses teks OCR
def preprocess_ingredients(ocr_text):
    combined_text = " ".join(ocr_text).lower()
    combined_text = " ".join(combined_text.split())
    combined_text = re.sub(r'\s*[;/]\s*', ', ', combined_text)
    combined_text = re.sub(r',\s*,', ',', combined_text)
    combined_text = combined_text.strip(', ')
    return combined_text

# Fungsi untuk memuat data bahan berbahaya
def load_hazardous_materials(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Bahan Berbahaya' not in df.columns or 'Analisis' not in df.columns:
            raise ValueError("File CSV harus memiliki kolom 'Bahan Berbahaya' dan 'Analisis'")
        return df
    except Exception as e:
        print(f"Error saat memuat file: {e}")
        return None

# Fungsi untuk memeriksa bahan berbahaya dalam teks input
def check_hazardous_materials(input_text, df):
    detected_materials = []
    for _, row in df.iterrows():
        if row['Bahan Berbahaya'].lower() in input_text.lower():
            detected_materials.append({
                'Bahan Berbahaya': row['Bahan Berbahaya'],
                'Analisis': row['Analisis']
            })
    return detected_materials

# Fungsi untuk prediksi jenis kulit berdasarkan teks input
def predict_skin_type(ingredient_texts, threshold=0.8, maxlen=100):
    try:
        # Tokenize dan pad teks input
        sequences = tokenizer.texts_to_sequences(ingredient_texts)
        padded_data = pad_sequences(sequences, padding='post', maxlen=maxlen)

        # Prediksi probabilitas untuk setiap jenis kulit
        predictions = model.predict(padded_data)

        # Terapkan threshold untuk multi-label classification
        predicted_labels = (predictions > threshold).astype(int)

        # Daftar skin types yang sesuai dengan urutan label
        skin_types = ['combination', 'dry', 'normal', 'oily', 'sensitive']

        # Menentukan skin types berdasarkan prediksi
        predicted_skin_types = [skin_types[i] for i in range(len(predicted_labels[0])) if predicted_labels[0][i] == 1]

        # Mengembalikan hasil dalam format JSON
        return {
            'predicted_skin_types': predicted_skin_types
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': str(e)}

# Route untuk menerima file gambar dan melakukan OCR
@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
        img_np = np.array(img)
        raw_result = reader.readtext(img_np, detail=0)
        processed_result = preprocess_ingredients(raw_result)
        return jsonify({'text': processed_result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route untuk melakukan prediksi berdasarkan teks input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ingredient_text = data.get('text', '')

        if not ingredient_text:
            return jsonify({'error': 'Text input is required'}), 400

        # Muat data bahan berbahaya
        file_path = "database.csv"
        df_bahan = load_hazardous_materials(file_path)

        if df_bahan is None:
            return jsonify({'error': 'Failed to load hazardous materials data'}), 500

        detected_materials = check_hazardous_materials(ingredient_text, df_bahan)

        if detected_materials:
            return jsonify({'hazardous_materials': detected_materials})
        else:
            prediction_result = predict_skin_type([ingredient_text])  # Mengambil hasil dari fungsi prediksi
            return jsonify(prediction_result)  # Mengembalikan hasil prediksi dalam format JSON

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
