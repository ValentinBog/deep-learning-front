from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime
import base64
from io import BytesIO
import uuid
import shutil

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

def get_patient_folder():
    """Obtiene o crea el directorio temporal del paciente actual"""
    if 'patient_id' not in session:
        session['patient_id'] = str(uuid.uuid4())
    
    patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['patient_id'])
    os.makedirs(patient_folder, exist_ok=True)
    return patient_folder

def cleanup_patient_folder():
    """Limpia el directorio temporal del paciente"""
    if 'patient_id' in session:
        patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['patient_id'])
        if os.path.exists(patient_folder):
            shutil.rmtree(patient_folder)
        session.pop('patient_id', None)

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
    
    # Asegurar que hay una sesión de paciente activa
    get_patient_folder()
    
    model_config = MODELS_CONFIG[model_id]
    return render_template('model_interface.html', 
                         model_id=model_id, 
                         model_config=model_config)

@app.route('/new_patient/<model_id>', methods=['POST'])
def new_patient(model_id):
    """Crea una nueva sesión de paciente limpiando el directorio anterior"""
    cleanup_patient_folder()
    get_patient_folder()  # Crear nuevo directorio
    return jsonify({'success': True, 'message': 'Nueva sesión de paciente creada'})

@app.route('/upload_image/<model_id>', methods=['POST'])
def upload_image(model_id):
    """Endpoint para subir múltiples imágenes sin hacer predicción"""
    if model_id not in MODELS_CONFIG:
        return jsonify({'error': 'Modelo no encontrado'}), 404
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Obtener directorio del paciente
            patient_folder = get_patient_folder()
            
            # Guardar archivo en el directorio del paciente
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Incluir milisegundos
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(patient_folder, filename)
            file.save(filepath)
            
            # Crear miniatura para vista previa
            image = cv2.imread(filepath)
            if image is None:
                os.remove(filepath)
                return jsonify({'error': 'No se pudo procesar la imagen'}), 400
            
            # Redimensionar para miniatura (manteniendo aspecto)
            height, width = image.shape[:2]
            max_size = 150
            if width > height:
                new_width = max_size
                new_height = int((max_size * height) / width)
            else:
                new_height = max_size
                new_width = int((max_size * width) / height)
            
            thumbnail = cv2.resize(image, (new_width, new_height))
            
            # Convertir a base64
            _, buffer = cv2.imencode('.jpg', thumbnail)
            thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'filename': filename,
                'thumbnail': thumbnail_base64,
                'message': 'Imagen cargada correctamente'
            })
            
        except Exception as e:
            return jsonify({'error': f'Error al procesar: {str(e)}'}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

@app.route('/get_uploaded_images/<model_id>')
def get_uploaded_images(model_id):
    """Obtiene la lista de imágenes cargadas para el paciente actual"""
    try:
        patient_folder = get_patient_folder()
        images = []
        
        for filename in os.listdir(patient_folder):
            if allowed_file(filename):
                filepath = os.path.join(patient_folder, filename)
                
                # Crear miniatura
                image = cv2.imread(filepath)
                if image is not None:
                    height, width = image.shape[:2]
                    max_size = 150
                    if width > height:
                        new_width = max_size
                        new_height = int((max_size * height) / width)
                    else:
                        new_height = max_size
                        new_width = int((max_size * width) / height)
                    
                    thumbnail = cv2.resize(image, (new_width, new_height))
                    _, buffer = cv2.imencode('.jpg', thumbnail)
                    thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    images.append({
                        'filename': filename,
                        'thumbnail': thumbnail_base64
                    })
        
        return jsonify({'success': True, 'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<model_id>', methods=['POST'])
def predict(model_id):
    """Endpoint para realizar predicciones en todas las imágenes del paciente"""
    if model_id not in MODELS_CONFIG:
        return jsonify({'error': 'Modelo no encontrado'}), 404
    
    try:
        patient_folder = get_patient_folder()
        image_files = [f for f in os.listdir(patient_folder) if allowed_file(f)]
        
        if not image_files:
            return jsonify({'error': 'No hay imágenes cargadas para el paciente'}), 400
        
        results_list = []
        model_config = MODELS_CONFIG[model_id]
        target_size = model_config['input_size']
        
        for filename in image_files:
            filepath = os.path.join(patient_folder, filename)
            
            try:
                # Preprocesar imagen
                processed_image, resized_image = preprocess_image(filepath, target_size)
                
                # Realizar predicción según el tipo de modelo
                if model_id == 'fibrosis_hepatica':
                    prediction_results = predict_fibrosis(processed_image)
                else:
                    continue  # Saltar si el modelo no está implementado
                
                # Convertir imagen procesada a base64 para mostrar
                _, buffer = cv2.imencode('.png', resized_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                results_list.append({
                    'filename': filename,
                    'results': prediction_results,
                    'processed_image': img_base64
                })
                
            except Exception as e:
                # Si hay error con una imagen, continuar con las otras
                results_list.append({
                    'filename': filename,
                    'error': f'Error al procesar {filename}: {str(e)}'
                })
                continue
        
        # Limpiar directorio del paciente después de la predicción
        cleanup_patient_folder()
        
        return jsonify({
            'success': True,
            'results': results_list,
            'model_name': model_config['name'],
            'total_images': len(results_list)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error al procesar: {str(e)}'}), 500@app.route('/api/models')
def get_models():
    """API endpoint para obtener información de los modelos disponibles"""
    return jsonify(MODELS_CONFIG)

if __name__ == '__main__':
    # Crear directorio de uploads si no existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
