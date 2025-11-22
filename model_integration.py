# model_integration.py - Exemple d'intégration de votre modèle réel

"""
Ce fichier montre comment intégrer votre modèle de deep learning entraîné
dans l'application web. Remplacez les fonctions de simulation par votre modèle réel.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
import torchvision.transforms as transforms
from PIL import Image

class FibrosisModel:
    """
    Classe pour encapsuler votre modèle de fibrosis hépatique.
    Adaptez cette classe selon votre framework (TensorFlow/PyTorch) et architecture.
    """
    
    def __init__(self, model_path, framework='tensorflow'):
        """
        Initialise le modèle.
        
        Args:
            model_path (str): Chemin vers le fichier du modèle sauvegardé
            framework (str): 'tensorflow' ou 'pytorch'
        """
        self.framework = framework
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle selon le framework utilisé."""
        try:
            if self.framework == 'tensorflow':
                # Pour TensorFlow/Keras
                self.model = load_model(self.model_path)
                print(f"✅ Modèle TensorFlow chargé depuis {self.model_path}")
                
            elif self.framework == 'pytorch':
                # Pour PyTorch
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                print(f"✅ Modèle PyTorch chargé depuis {self.model_path}")
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            raise
    
    def preprocess_image(self, image_array, target_size=(256, 192)):
        """
        Préprocesse l'image pour la prédiction.
        
        Args:
            image_array (np.ndarray): Image d'entrée
            target_size (tuple): Taille cible (largeur, hauteur)
            
        Returns:
            np.ndarray: Image préprocessée
        """
        try:
            # Redimensionner l'image
            resized = cv2.resize(image_array, target_size)
            
            if self.framework == 'tensorflow':
                # Normalisation pour TensorFlow (0-1)
                normalized = resized.astype(np.float32) / 255.0
                # Ajouter dimension batch
                processed = np.expand_dims(normalized, axis=0)
                
            elif self.framework == 'pytorch':
                # Conversion pour PyTorch
                pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                processed = transform(pil_image).unsqueeze(0)
            
            return processed
            
        except Exception as e:
            raise Exception(f"Erreur lors du préprocessing: {e}")
    
    def predict(self, image_array):
        """
        Effectue la prédiction sur une image.
        
        Args:
            image_array (np.ndarray): Image d'entrée
            
        Returns:
            dict: Résultats de prédiction structurés
        """
        try:
            # Préprocessing
            processed_image = self.preprocess_image(image_array)
            
            if self.framework == 'tensorflow':
                # Prédiction TensorFlow
                predictions = self.model.predict(processed_image, verbose=0)
                
                # Si votre modèle a plusieurs sorties (multiclass + binary)
                if isinstance(predictions, list):
                    multiclass_pred = predictions[0]  # Première sortie: F0-F4
                    binary_pred_1 = predictions[1]    # Deuxième sortie: F0 vs F1-F4
                    binary_pred_2 = predictions[2] if len(predictions) > 2 else None  # F0-F1 vs F2-F4
                else:
                    # Un seul output multiclass
                    multiclass_pred = predictions
                
            elif self.framework == 'pytorch':
                # Prédiction PyTorch
                with torch.no_grad():
                    outputs = self.model(processed_image)
                    
                    if isinstance(outputs, tuple):
                        multiclass_pred = torch.softmax(outputs[0], dim=1).cpu().numpy()
                        binary_pred_1 = torch.sigmoid(outputs[1]).cpu().numpy() if len(outputs) > 1 else None
                        binary_pred_2 = torch.sigmoid(outputs[2]).cpu().numpy() if len(outputs) > 2 else None
                    else:
                        multiclass_pred = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # Structurer les résultats
            return self._structure_results(multiclass_pred, binary_pred_1, binary_pred_2)
            
        except Exception as e:
            raise Exception(f"Erreur lors de la prédiction: {e}")
    
    def _structure_results(self, multiclass_pred, binary_pred_1=None, binary_pred_2=None):
        """
        Structure les résultats de prédiction dans le format attendu par l'interface.
        
        Args:
            multiclass_pred (np.ndarray): Probabilités pour chaque classe F0-F4
            binary_pred_1 (np.ndarray): Probabilités F0 vs F1-F4
            binary_pred_2 (np.ndarray): Probabilités F0-F1 vs F2-F4
            
        Returns:
            dict: Résultats structurés
        """
        # Classes de fibrosis
        fibrosis_classes = ['F0', 'F1', 'F2', 'F3', 'F4']
        
        # Extraction des probabilités multiclass
        if len(multiclass_pred.shape) > 1:
            probs = multiclass_pred[0]  # Première image du batch
        else:
            probs = multiclass_pred
        
        # Création du dictionnaire de probabilités
        probabilities = {fibrosis_classes[i]: float(probs[i]) for i in range(len(probs))}
        
        # Classe prédite (plus haute probabilité)
        predicted_class = fibrosis_classes[np.argmax(probs)]
        
        # Résultats multiclass
        multiclass_results = {
            'predicted_class': predicted_class,
            'probabilities': probabilities
        }
        
        # Résultats binaires
        binary_results = []
        
        # Première classification binaire: F0 vs F1-F4
        if binary_pred_1 is not None:
            f0_prob = float(binary_pred_1[0]) if len(binary_pred_1.shape) > 1 else float(binary_pred_1)
            f1_f4_prob = 1 - f0_prob
        else:
            # Calculer à partir des probabilités multiclass
            f0_prob = probabilities['F0']
            f1_f4_prob = sum([probabilities[cls] for cls in ['F1', 'F2', 'F3', 'F4']])
        
        binary_results.append({
            'name': 'Detección de Fibrosis',
            'predictions': {
                'F0 (Sin fibrosis)': f0_prob,
                'F1-F4 (Con fibrosis)': f1_f4_prob
            },
            'predicted_class': 'F0 (Sin fibrosis)' if f0_prob > f1_f4_prob else 'F1-F4 (Con fibrosis)'
        })
        
        # Deuxième classification binaire: F0-F1 vs F2-F4
        if binary_pred_2 is not None:
            f0_f1_prob = float(binary_pred_2[0]) if len(binary_pred_2.shape) > 1 else float(binary_pred_2)
            f2_f4_prob = 1 - f0_f1_prob
        else:
            # Calculer à partir des probabilités multiclass
            f0_f1_prob = probabilities['F0'] + probabilities['F1']
            f2_f4_prob = sum([probabilities[cls] for cls in ['F2', 'F3', 'F4']])
        
        binary_results.append({
            'name': 'Fibrosis Significativa',
            'predictions': {
                'F0-F1 (Leve)': f0_f1_prob,
                'F2-F4 (Significativa)': f2_f4_prob
            },
            'predicted_class': 'F0-F1 (Leve)' if f0_f1_prob > f2_f4_prob else 'F2-F4 (Significativa)'
        })
        
        return {
            'multiclass': multiclass_results,
            'binary': binary_results
        }


# Fonction d'interface pour intégrer dans app.py
def predict_fibrosis_real(image_array):
    """
    Fonction à utiliser dans app.py pour remplacer la simulation.
    
    Args:
        image_array (np.ndarray): Image préprocessée
        
    Returns:
        dict: Résultats de prédiction
    """
    try:
        # Initialiser le modèle (vous pouvez le faire une seule fois au démarrage de l'app)
        model_path = 'models/fibrosis_model.h5'  # Ajustez le chemin
        fibrosis_model = FibrosisModel(model_path, framework='tensorflow')  # ou 'pytorch'
        
        # Effectuer la prédiction
        results = fibrosis_model.predict(image_array)
        
        return results
        
    except Exception as e:
        print(f"Erreur dans predict_fibrosis_real: {e}")
        # Retourner une erreur ou des résultats par défaut
        raise


# Exemple d'utilisation avec un modèle de segmentation + classification
class FibrosisSegmentationModel:
    """
    Exemple pour un modèle en deux étapes: segmentation puis classification
    (comme décrit dans votre document: U-Net + 1D CNN)
    """
    
    def __init__(self, segmentation_model_path, classification_model_path):
        """
        Initialise les deux modèles.
        
        Args:
            segmentation_model_path (str): Chemin vers le modèle de segmentation (U-Net)
            classification_model_path (str): Chemin vers le modèle de classification (1D CNN)
        """
        self.segmentation_model = load_model(segmentation_model_path)
        self.classification_model = load_model(classification_model_path)
        
        print("✅ Modèles de segmentation et classification chargés")
    
    def segment_liver(self, image):
        """Segmente la région du foie dans l'image."""
        # Préprocessing pour segmentation
        processed = cv2.resize(image, (256, 192))
        processed = processed.astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        # Prédiction de segmentation
        mask = self.segmentation_model.predict(processed, verbose=0)
        
        return mask[0]  # Retourner le masque
    
    def extract_features(self, image, mask):
        """Extrait les features pour la classification 1D CNN."""
        # Appliquer le masque
        masked_image = image * mask
        
        # Extraire des features (exemple: spectres de fréquence comme dans l'article)
        # Ceci dépend de votre implémentation spécifique
        
        # Exemple simplifié: moyennes par canal
        features = np.mean(masked_image.reshape(-1, 3), axis=0)
        
        return features
    
    def classify_fibrosis(self, features):
        """Classifie le stade de fibrosis à partir des features."""
        # Adapter selon votre architecture 1D CNN
        features = features.reshape(1, -1)  # Adapter la forme
        
        predictions = self.classification_model.predict(features, verbose=0)
        
        return predictions
    
    def predict(self, image_array):
        """Pipeline complet: segmentation + classification."""
        try:
            # Étape 1: Segmentation
            mask = self.segment_liver(image_array)
            
            # Étape 2: Extraction de features
            features = self.extract_features(image_array, mask)
            
            # Étape 3: Classification
            predictions = self.classify_fibrosis(features)
            
            # Structurer les résultats
            return self._structure_results(predictions)
            
        except Exception as e:
            raise Exception(f"Erreur dans le pipeline de prédiction: {e}")


# Instructions pour remplacer dans app.py:
"""
1. Remplacez la fonction predict_fibrosis() dans app.py par:

from model_integration import predict_fibrosis_real

def predict_fibrosis(image_array):
    return predict_fibrosis_real(image_array)

2. Ou pour le modèle en deux étapes:

from model_integration import FibrosisSegmentationModel

# Initialiser une seule fois au démarrage
segmentation_model = FibrosisSegmentationModel(
    'models/segmentation_model.h5',
    'models/classification_model.h5'
)

def predict_fibrosis(image_array):
    return segmentation_model.predict(image_array)
"""
