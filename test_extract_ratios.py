#!/usr/bin/env python3
# Test script pour extraire les ratios de masques

# Exemple de sortie raw du modèle
raw_output = '''
📂 Paciente: /path/to/patient/folder
   Imágenes encontradas: 17
 - Procesando IM-0001-0018.jpg...
✅ Imagen válida para hígado (ratio máscara = 0.2131)
 - Procesando IM-0001-0019.jpg...
✅ Imagen válida para hígado (ratio máscara = 0.2342)
 - Procesando IM-0001-0020.jpg...
✅ Imagen válida para hígado (ratio máscara = 0.2483)
 - Procesando IM-0001-0047.jpg...
✅ Imagen válida para hígado (ratio máscara = 0.1804)
'''

def extract_image_ratios(raw_output):
    """Extrae los ratios de máscara de la salida raw del modelo"""
    raw_lines = raw_output.split('\n')
    image_ratios = {}
    current_filename = None
    
    for line in raw_lines:
        print(f"Procesando línea: {line.strip()}")
        
        # Buscar línea de procesamiento
        if 'Procesando' in line and '.jpg' in line:
            import re
            filename_match = re.search(r'(IM-\d+-\d+\.jpg)', line)
            if filename_match:
                current_filename = filename_match.group(1)
                print(f"  -> Encontrado archivo: {current_filename}")
        
        # Buscar línea de ratio inmediatamente después
        elif current_filename and 'ratio máscara =' in line:
            import re
            ratio_match = re.search(r'ratio máscara = ([\d.]+)', line)
            if ratio_match:
                ratio = float(ratio_match.group(1))
                image_ratios[current_filename] = ratio
                print(f"  -> Ratio para {current_filename}: {ratio}")
                current_filename = None  # Reset para la siguiente imagen
    
    return image_ratios

# Test
ratios = extract_image_ratios(raw_output)
print(f"\nRatios extraídos: {ratios}")
