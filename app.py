from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
import base64
from io import BytesIO

# Configuration
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Extensiones de archivo permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# Configuration des modèles disponibles
MODELS_CONFIG = {
    'fibrosis_hepatica': {
        'name': 'Detección de Fibrosis Hepática',
        'description': 'Clasificación de estadios de fibrosis hepática (F0-F4) mediante ecografía',
        'input_size': (256, 192),
        'original_size': (640, 480),
        'categories': {
            'F0': 'Sin fibrosis - Tejido hepático sano',
            'F1': 'Fibrosis portal - Cicatrices alrededor de las venas portales',
            'F2': 'Fibrosis periportal - Extensión hacia zonas periféricas', 
            'F3': 'Fibrosis septal - Bandas fibrosas que dividen el parénquima',
            'F4': 'Cirrosis - Reemplazo del tejido funcional por tejido cicatricial'
        },
        'binary_classifications': [
            {'name': 'Detección de Fibrosis', 'classes': ['F0 (Sin fibrosis)', 'F1-F4 (Con fibrosis)']},
            {'name': 'Fibrosis Significativa', 'classes': ['F0-F1 (Leve)', 'F2-F4 (Significativa)']}
        ]
    }
    # Aquí se pueden agregar más modelos en el futuro
}

def allowed_file(filename):
    """Verifica si la extensión del archivo está permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size):
    """Preprocesa la imagen para el modelo"""
    try:
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("No se pudo cargar la imagen")
        
        # Redimensionar
        resized = cv2.resize(image, target_size)
        
        # Normalizar
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized, resized
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {str(e)}")

def predict_fibrosis(image_array):
    """
    Simula la predicción del modelo de fibrosis
    Reemplazar con tu modelo real
    """
    # Simulación de predicción
    # Reemplazar esto con la carga y predicción de tu modelo real
    
    import random
    np.random.seed(42)
    
    # Simulación de probabilidades para cada categoría
    probabilities = {
        'F0': random.uniform(0.1, 0.9),
        'F1': random.uniform(0.1, 0.8),
        'F2': random.uniform(0.1, 0.7),
        'F3': random.uniform(0.1, 0.6),
        'F4': random.uniform(0.1, 0.5)
    }
    
    # Normalizar probabilidades
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    
    # Predicción principal (clase con mayor probabilidad)
    predicted_class = max(probabilities, key=probabilities.get)
    
    # Clasificaciones binarias
    f0_prob = probabilities['F0']
    f1_f4_prob = 1 - f0_prob
    
    f0_f1_prob = probabilities['F0'] + probabilities['F1']
    f2_f4_prob = 1 - f0_f1_prob
    
    binary_predictions = [
        {
            'name': 'Detección de Fibrosis',
            'predictions': {
                'F0 (Sin fibrosis)': f0_prob,
                'F1-F4 (Con fibrosis)': f1_f4_prob
            },
            'predicted_class': 'F0 (Sin fibrosis)' if f0_prob > f1_f4_prob else 'F1-F4 (Con fibrosis)'
        },
        {
            'name': 'Fibrosis Significativa', 
            'predictions': {
                'F0-F1 (Leve)': f0_f1_prob,
                'F2-F4 (Significativa)': f2_f4_prob
            },
            'predicted_class': 'F0-F1 (Leve)' if f0_f1_prob > f2_f4_prob else 'F2-F4 (Significativa)'
        }
    ]
    
    return {
        'multiclass': {
            'predicted_class': predicted_class,
            'probabilities': probabilities
        },
        'binary': binary_predictions
    }

@app.route('/')
def index():
    """Página principal con selección de modelos"""
    return render_template('index.html', models=MODELS_CONFIG)

@app.route('/model/<model_id>')
def model_interface(model_id):
    """Interface para un modelo específico"""
    if model_id not in MODELS_CONFIG:
        flash('Modelo no encontrado', 'error')
        return redirect(url_for('index'))
    
    model_config = MODELS_CONFIG[model_id]
    return render_template('model_interface.html', 
                         model_id=model_id, 
                         model_config=model_config)

@app.route('/predict/<model_id>', methods=['POST'])
def predict(model_id):
    """Endpoint para realizar predicciones"""
    if model_id not in MODELS_CONFIG:
        return jsonify({'error': 'Modelo no encontrado'}), 404
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Guardar archivo
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Obtener configuración del modelo
            model_config = MODELS_CONFIG[model_id]
            target_size = model_config['input_size']
            
            # Preprocesar imagen
            processed_image, resized_image = preprocess_image(filepath, target_size)
            
            # Realizar predicción según el tipo de modelo
            if model_id == 'fibrosis_hepatica':
                results = predict_fibrosis(processed_image)
            else:
                return jsonify({'error': 'Tipo de modelo no implementado'}), 500
            
            # Convertir imagen procesada a base64 para mostrar
            _, buffer = cv2.imencode('.png', resized_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Limpiar archivo temporal
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'results': results,
                'processed_image': img_base64,
                'model_name': model_config['name']
            })
            
        except Exception as e:
            # Limpiar archivo si existe
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error al procesar: {str(e)}'}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

@app.route('/api/models')
def get_models():
    """API endpoint para obtener información de los modelos disponibles"""
    return jsonify(MODELS_CONFIG)

if __name__ == '__main__':
    # Crear directorio de uploads si no existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
