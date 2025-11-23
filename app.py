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
from model_wrapper import run_liver_fibrosis_model
from models_config import MODELS_CONFIG

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
        
        # Ejecutar el modelo real usando el wrapper
        print(f"Ejecutando modelo en carpeta: {patient_folder}")
        model_result = run_liver_fibrosis_model(patient_folder)
        
        # Verificar si hubo errores en el modelo
        if 'error' in model_result:
            print(f"Error del modelo: {model_result['error']}")
            # Si hay error, mostrar la salida raw para debugging
            if 'raw_output' in model_result:
                print("Salida raw del modelo:")
                print(model_result['raw_output'])
            
            return jsonify({
                'error': f"Error en el modelo: {model_result.get('error', 'Error desconocido')}",
                'details': model_result.get('raw_output', 'Sin detalles adicionales')
            }), 500
        
        # Estructurar los resultados para la interfaz web
        structured_result = {
            'status': 'success',
            'patient_info': {
                'path': model_result.get('patient_folder', patient_folder),
                'images_found': model_result.get('total_images', len(image_files))
            },
            'summary': {
                'total_images': model_result.get('total_images', 0),
                'used_images': model_result.get('used_images', 0),
                'discarded_images': model_result.get('discarded_images', 0),
                'stage_counts': model_result.get('per_class_counts', {}),
                'probabilities': model_result.get('patient_probs', []),
                'probability_by_stage': {}
            },
            'final_stage': model_result.get('patient_stage', 'Desconocido'),
            'processed_images': []
        }
        
        # Crear diccionario de probabilidades por etapa
        if model_result.get('patient_probs'):
            stages = ['F0', 'F1', 'F2', 'F3', 'F4']
            probs = model_result['patient_probs']
            structured_result['summary']['probability_by_stage'] = {
                stages[i]: probs[i] for i in range(min(len(stages), len(probs)))
            }
        
        # Procesar resultados por imagen y extraer ratios de máscara
        if 'per_image_results' in model_result:
            # Extraer ratios de máscara de la salida raw
            image_ratios = {}
            if 'raw_output' in model_result:
                raw_lines = model_result['raw_output'].split('\n')
                current_filename = None
                
                for line in raw_lines:
                    # Buscar línea de procesamiento
                    if 'Procesando' in line and '.jpg' in line:
                        import re
                        filename_match = re.search(r'(IM-\d+-\d+\.jpg)', line)
                        if filename_match:
                            current_filename = filename_match.group(1)
                    
                    # Buscar línea de ratio inmediatamente después
                    elif current_filename and 'ratio máscara =' in line:
                        import re
                        ratio_match = re.search(r'ratio máscara = ([\d.]+)', line)
                        if ratio_match:
                            ratio = float(ratio_match.group(1))
                            image_ratios[current_filename] = ratio
                            current_filename = None  # Reset para la siguiente imagen
                
                print(f"Ratios extraídos: {image_ratios}")  # DEBUG
            
            for img_result in model_result['per_image_results']:
                filename = img_result.get('filename', 'desconocido')
                ratio_value = image_ratios.get(filename, None)
                print(f"Imagen {filename}: ratio = {ratio_value}")  # DEBUG
                
                processed_img = {
                    'name': filename,
                    'valid': img_result.get('used', False),
                    'pred_class': img_result.get('pred_class', 'N/A'),
                    'probs': img_result.get('probs', []),
                    'ratio': ratio_value  # Agregar el ratio extraído
                }
                structured_result['processed_images'].append(processed_img)
        
        # Incluir salida raw para debugging (opcional, puedes comentar esta línea en producción)
        structured_result['debug'] = {
            'raw_output': model_result.get('raw_output', ''),
            'error_output': model_result.get('error_output', '')
        }
        
        # Limpiar directorio del paciente después de la predicción
        cleanup_patient_folder()
        
        return jsonify(structured_result)
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error en predict: {str(e)}")
        print(error_traceback)
        return jsonify({
            'error': f'Error al procesar: {str(e)}',
            'traceback': error_traceback
        }), 500

@app.route('/api/models')
def get_models():
    """API endpoint para obtener información de los modelos disponibles"""
    return jsonify(MODELS_CONFIG)
def get_models():
    """API endpoint para obtener información de los modelos disponibles"""
    return jsonify(MODELS_CONFIG)

if __name__ == '__main__':
    # Crear directorio de uploads si no existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
