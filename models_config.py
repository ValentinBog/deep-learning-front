# Configuration des modèles - models_config.py

"""
Configuration centralisée pour tous les modèles disponibles dans le système.
Ce fichier permet d'ajouter facilement de nouveaux modèles sans modifier le code principal.
"""

MODELS_CONFIG = {
    # Modèle de détection de fibrosis hépatique
    'fibrosis_hepatica': {
        'name': 'Detección de Fibrosis Hepática',
        'description': 'Clasificación automática de estadios de fibrosis hepática (F0-F4) mediante análisis de imágenes de ecografía con deep learning',
        'version': '1.0',
        'author': 'Universidad de los Andes - Grupo de Investigación en IA Médica',
        
        # Configuración técnica
        'input_size': (256, 192),        # Tamaño de procesamiento del modelo
        'original_size': (640, 480),     # Tamaño original recomendado
        'channels': 3,                   # RGB
        'model_type': 'cnn_2d_1d',      # Tipo de arquitectura
        'model_path': 'models/fibrosis_model.h5',  # Ruta del modelo entrenado
        
        # Categorías de diagnóstico (multiclase)
        'categories': {
            'F0': 'Sin fibrosis - Tejido hepático completamente sano',
            'F1': 'Fibrosis portal - Aparición de cicatrices alrededor de las venas portales',
            'F2': 'Fibrosis periportal - Extensión de las lesiones hacia las zonas periféricas',
            'F3': 'Fibrosis septal - Formación de bandas fibrosas que dividen el parénquima hepático',
            'F4': 'Cirrosis - Reemplazo del tejido funcional por tejido cicatricial'
        },
        
        # Clasificaciones binarias disponibles
        'binary_classifications': [
            {
                'name': 'Detección de Fibrosis',
                'description': 'Diferencia entre ausencia y presencia de fibrosis',
                'classes': ['F0 (Sin fibrosis)', 'F1-F4 (Con fibrosis)'],
                'threshold': 0.5
            },
            {
                'name': 'Fibrosis Significativa', 
                'description': 'Identifica fibrosis clínicamente significativa',
                'classes': ['F0-F1 (Leve o ausente)', 'F2-F4 (Significativa)'],
                'threshold': 0.5
            }
        ],
        
        # Métricas de rendimiento (para mostrar en la interfaz)
        'performance': {
            'accuracy': 0.89,
            'sensitivity': 0.91,
            'specificity': 0.87,
            'auc_roc': 0.92
        },
        
        # Información clínica
        'clinical_info': {
            'indication': 'Pacientes con sospecha de enfermedad hepática crónica',
            'contraindications': 'Imágenes de baja calidad o artefactos significativos',
            'limitations': 'Este sistema es una herramienta de apoyo diagnóstico y no reemplaza el criterio médico'
        }
    },
    
    # Plantilla para futuros modelos
    'template_nuevo_modelo': {
        'name': 'Nombre del Nuevo Modelo',
        'description': 'Descripción detallada del modelo y su propósito clínico',
        'version': '1.0',
        'author': 'Autor/Institución',
        
        # Configuración técnica
        'input_size': (224, 224),
        'original_size': (512, 512),
        'channels': 3,
        'model_type': 'cnn_2d',
        'model_path': 'models/nuevo_modelo.h5',
        
        # Categorías (ajustar según el caso)
        'categories': {
            'Normal': 'Tejido normal sin patología',
            'Anormal': 'Presencia de patología'
        },
        
        # Clasificaciones binarias
        'binary_classifications': [
            {
                'name': 'Clasificación Principal',
                'description': 'Descripción de la clasificación',
                'classes': ['Normal', 'Anormal'],
                'threshold': 0.5
            }
        ],
        
        # Métricas
        'performance': {
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'auc_roc': 0.0
        },
        
        # Información clínica
        'clinical_info': {
            'indication': 'Indicaciones clínicas',
            'contraindications': 'Contraindicaciones',
            'limitations': 'Limitaciones del modelo'
        }
    }
}

# Configuración general del sistema
SYSTEM_CONFIG = {
    'app_name': 'Sistema de Diagnóstico por IA',
    'version': '1.0.0',
    'institution': 'Universidad de los Andes, Bogotá, Colombia',
    'contact': 'Grupo de Investigación en IA Médica',
    
    # Configuración de archivos
    'max_file_size_mb': 16,
    'allowed_extensions': ['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    'upload_folder': 'uploads',
    
    # Configuración de seguridad
    'secret_key': 'your-secret-key-change-in-production',
    'session_timeout': 3600,  # 1 hora en segundos
    
    # Configuración de logging
    'log_level': 'INFO',
    'log_file': 'logs/app.log'
}
