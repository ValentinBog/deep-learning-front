# 🔧 Guía de Integración del Modelo de Deep Learning

## 📍 Ubicación de archivos del modelo

### 1. **Archivos del modelo entrenado**
Coloque sus archivos de modelo en la carpeta `models/`:

```
web_app/
├── models/
│   ├── fibrosis_model.h5          # Modelo TensorFlow/Keras principal
│   ├── fibrosis_weights.h5        # Pesos del modelo (si están separados)
│   ├── model_architecture.json    # Arquitectura del modelo (si está separada)
│   ├── segmentation_model.h5      # Modelo de segmentación (Etapa 1)
│   ├── classification_model.h5    # Modelo de clasificación (Etapa 2)
│   └── preprocessing_config.json  # Configuración de preprocesamiento
```

### 2. **Para diferentes frameworks**

#### **Si usa TensorFlow/Keras:**
```bash
# Copie su modelo .h5 a la carpeta models/
cp su_modelo_entrenado.h5 web_app/models/fibrosis_model.h5
```

#### **Si usa PyTorch:**
```bash
# Copie su modelo .pth/.pt a la carpeta models/
cp su_modelo_entrenado.pth web_app/models/fibrosis_model.pth
```

#### **Si usa ONNX:**
```bash
# Copie su modelo .onnx a la carpeta models/
cp su_modelo_entrenado.onnx web_app/models/fibrosis_model.onnx
```

---

## 🔧 Pasos para la integración

### **Paso 1: Actualizar la configuración**

Edite `models_config.py` línea 17:

```python
# Cambie esta línea:
'model_path': 'models/fibrosis_model.h5',

# Por la ruta real de su modelo:
'model_path': 'models/SU_MODELO_REAL.h5',
```

### **Paso 2: Implementar su modelo real**

Edite `model_integration.py` y reemplace las funciones de simulación:

#### **Para TensorFlow/Keras:**

```python
class FibrosisModel:
    def __init__(self, model_path):
        self.segmentation_model = load_model('models/segmentation_model.h5')
        self.classification_model = load_model('models/classification_model.h5')
    
    def predict(self, image_array):
        # Etapa 1: Segmentación
        segmented = self.segmentation_model.predict(image_array)
        
        # Etapa 2: Clasificación
        prediction = self.classification_model.predict(segmented)
        
        return self.format_results(prediction)
```

#### **Para PyTorch:**

```python
import torch
import torch.nn as nn

class FibrosisModel:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
    
    def predict(self, image_array):
        with torch.no_grad():
            tensor_image = self.preprocess_for_pytorch(image_array)
            prediction = self.model(tensor_image)
            return self.format_results(prediction)
```

### **Paso 3: Configurar el preprocesamiento**

Según su documento, sus imágenes necesitan:
- **Entrada original:** 640×480
- **Procesamiento:** 256×192 (factor 0.4)
- **Augmentación:** rotaciones (0°, 90°, 180°, 270°)

```python
def preprocess_image(self, image_array):
    """Preprocesamiento específico para fibrosis hepática"""
    # Redimensionar de 640×480 a 256×192
    resized = cv2.resize(image_array, (256, 192))
    
    # Normalización
    normalized = resized.astype(np.float32) / 255.0
    
    # Agregar dimensión batch
    return np.expand_dims(normalized, axis=0)
```

---

## 🎯 Integración específica para su proyecto

### **Arquitectura de dos etapas (según su documento)**

```python
class FibrosisPipeline:
    def __init__(self):
        # Etapa 1: Segmentación (U-Net/Attention U-Net)
        self.segmentation_model = load_model('models/segmentation_unet.h5')
        
        # Etapa 2: Clasificación (1D CNN)
        self.classification_model = load_model('models/classification_1dcnn.h5')
    
    def predict_fibrosis(self, ultrasound_image):
        """Pipeline completo de predicción"""
        
        # 1. Preprocesar imagen (640×480 → 256×192)
        preprocessed = self.preprocess_ultrasound(ultrasound_image)
        
        # 2. Segmentar región del hígado
        liver_mask = self.segmentation_model.predict(preprocessed)
        
        # 3. Extraer región de interés
        liver_region = self.extract_liver_region(preprocessed, liver_mask)
        
        # 4. Clasificar estadio de fibrosis
        fibrosis_prediction = self.classification_model.predict(liver_region)
        
        # 5. Formatear resultados
        return self.format_clinical_results(fibrosis_prediction)
    
    def format_clinical_results(self, prediction):
        """Formatear resultados clínicos"""
        # Clasificación multiclase (F0-F4)
        class_probs = prediction[0]
        predicted_class = np.argmax(class_probs)
        confidence = np.max(class_probs)
        
        # Clasificaciones binarias
        f0_vs_rest = class_probs[0] vs sum(class_probs[1:])
        f01_vs_f234 = sum(class_probs[:2]) vs sum(class_probs[2:])
        
        return {
            'fibrosis_stage': f'F{predicted_class}',
            'confidence': float(confidence),
            'stage_probabilities': {
                'F0': float(class_probs[0]),
                'F1': float(class_probs[1]),
                'F2': float(class_probs[2]),
                'F3': float(class_probs[3]),
                'F4': float(class_probs[4])
            },
            'binary_classifications': {
                'no_fibrosis_vs_fibrosis': {
                    'probability': float(1 - class_probs[0]),
                    'classification': 'Fibrosis detectada' if class_probs[0] < 0.5 else 'Sin fibrosis'
                },
                'mild_vs_significant': {
                    'probability': float(sum(class_probs[2:])),
                    'classification': 'Fibrosis significativa' if sum(class_probs[2:]) > 0.5 else 'Fibrosis leve o ausente'
                }
            }
        }
```

---

## 📋 Lista de verificación para la integración

### ✅ **Antes de integrar:**

1. **Preparar archivos del modelo:**
   - [ ] Modelo de segmentación entrenado
   - [ ] Modelo de clasificación entrenado
   - [ ] Archivos de configuración/pesos
   - [ ] Script de preprocesamiento

2. **Verificar dependencias:**
   - [ ] TensorFlow/PyTorch instalado
   - [ ] OpenCV para procesamiento de imágenes
   - [ ] NumPy, PIL para manipulación de arrays

3. **Probar el modelo por separado:**
   - [ ] Cargar modelo correctamente
   - [ ] Hacer predicción con imagen de prueba
   - [ ] Verificar formato de salida

### ✅ **Durante la integración:**

1. **Actualizar configuración:**
   - [ ] Modificar `models_config.py`
   - [ ] Ajustar rutas de archivos
   - [ ] Configurar métricas de rendimiento

2. **Implementar funciones:**
   - [ ] Reemplazar `simulate_fibrosis_prediction()` en `model_integration.py`
   - [ ] Implementar preprocesamiento correcto
   - [ ] Manejar errores y excepciones

3. **Probar integración:**
   - [ ] Subir imagen de prueba
   - [ ] Verificar resultados
   - [ ] Comprobar tiempos de respuesta

---

## 🚀 Comandos rápidos para probar

```bash
# 1. Copiar su modelo
cp /ruta/a/su/modelo.h5 web_app/models/fibrosis_model.h5

# 2. Instalar dependencias adicionales si es necesario
pip install tensorflow opencv-python pillow

# 3. Probar el modelo
cd web_app
python -c "
from model_integration import FibrosisModel
model = FibrosisModel('models/fibrosis_model.h5')
print('✅ Modelo cargado correctamente')
"

# 4. Ejecutar aplicación
python app.py
```

---

## 🔍 Debugging común

### **Error: "No se puede cargar el modelo"**
- Verificar que el archivo existe en `models/`
- Comprobar compatibilidad de versiones TensorFlow/PyTorch
- Revisar formato del archivo (.h5, .pth, .onnx)

### **Error: "Dimensiones incorrectas"**
- Verificar preprocesamiento de imagen (256×192)
- Comprobar formato de entrada del modelo
- Revisar número de canales (RGB vs grayscale)

### **Predicciones incorrectas**
- Verificar normalización de imágenes
- Comprobar orden de clases (F0, F1, F2, F3, F4)
- Revisar umbral de clasificación binaria

---

¡Su modelo estará listo para usar en la interfaz web! 🎉
