#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper para integrar el modelo de fibrosis hepática en la aplicación web Flask.
"""

import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_liver_fibrosis_model(patient_folder_path):
    """
    Ejecuta el modelo de fibrosis hepática sobre una carpeta de paciente.
    
    Args:
        patient_folder_path (str): Ruta absoluta a la carpeta del paciente con imágenes
        
    Returns:
        dict: Resultado estructurado con la predicción del modelo
    """
    try:
        # Crear un script temporal que importa cargarModelo.py y ejecuta la función
        script_content = f'''
import sys
sys.path.append("{os.path.dirname(os.path.abspath(__file__))}")

# Importar todo lo necesario del script cargarModelo.py
from cargarModelo import infer_patient_folder

# Ejecutar la función de inferencia
resultado = infer_patient_folder(
    r"{patient_folder_path}",
    min_ratio=0.02,
    max_ratio=0.90,
    verbose=True
)

print("RESULTADO_JSON_START")
import json
print(json.dumps({{
    "patient_folder": "{patient_folder_path}",
    "total_images": resultado["total_images"],
    "used_images": resultado["used_images"],
    "discarded_images": resultado["discarded_images"],
    "per_class_counts": resultado["per_class_counts"],
    "patient_probs": resultado["patient_probs"].tolist() if resultado["patient_probs"] is not None else None,
    "patient_stage": resultado["patient_stage"],
    "per_image_results": resultado["per_image_results"]
}}, indent=2))
print("RESULTADO_JSON_END")

print("\\nEtapa final estimada para el paciente:", resultado["patient_stage"])
'''

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_script_path = temp_file.name

        try:
            # Ejecutar el script
            result = subprocess.run(
                [sys.executable, temp_script_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            output = result.stdout
            error_output = result.stderr
            
            # Buscar el JSON en la salida
            json_start = output.find("RESULTADO_JSON_START")
            json_end = output.find("RESULTADO_JSON_END")
            
            if json_start != -1 and json_end != -1:
                json_start += len("RESULTADO_JSON_START\n")
                json_data = output[json_start:json_end].strip()
                
                import json
                structured_result = json.loads(json_data)
                
                # Agregar la salida completa para debugging
                structured_result['raw_output'] = output
                structured_result['error_output'] = error_output
                
                return structured_result
            else:
                # Si no encontramos el JSON, devolvemos la salida raw
                return {
                    "error": "No se pudo extraer resultado estructurado",
                    "raw_output": output,
                    "error_output": error_output,
                    "return_code": result.returncode
                }
                
        finally:
            # Limpiar archivo temporal
            try:
                os.unlink(temp_script_path)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        return {
            "error": "Timeout: El modelo tardó más de 5 minutos en ejecutarse",
            "timeout": True
        }
    except Exception as e:
        return {
            "error": f"Error ejecutando el modelo: {str(e)}",
            "exception": True
        }

def test_model_wrapper():
    """
    Función de prueba para verificar que el wrapper funciona
    """
    test_folder = "/home/valentin/Bureau/DEEP LEARNING PROJET/FRONTGIT/deep-learning-front/uploads/4-7"
    if os.path.exists(test_folder):
        print("Probando el wrapper del modelo...")
        result = run_liver_fibrosis_model(test_folder)
        print("Resultado:")
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Carpeta de prueba no encontrada: {test_folder}")

if __name__ == "__main__":
    test_model_wrapper()
