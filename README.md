# Sistema de Diagnóstico por IA - Interfaz Web

Sistema web para la clasificación automática de fibrosis hepática mediante análisis de imágenes de ecografía utilizando deep learning.

## Características

- **Interface intuitive en espagnol** : Navigation et utilisation complètement en espagnol
- **Architecture scalable** : Possibilité d'ajouter facilement de nouveaux modèles
- **Traitement d'images** : Redimensionnement automatique aux dimensions requises (256×192)
- **Résultats détaillés** : Classifications multiclasses et binaires avec probabilités
- **Design responsive** : Compatible avec tous les appareils (desktop, tablette, mobile)

## Structure du projet

```
web_app/
├── app.py                 # Serveur Flask principal
├── requirements.txt       # Dépendances Python
├── README.md             # Ce fichier
├── static/
│   └── css/
│       └── style.css     # Styles CSS personnalisés
├── templates/
│   ├── index.html        # Page d'accueil
│   └── model_interface.html  # Interface du modèle
├── models/               # Dossier pour vos modèles entraînés
├── uploads/              # Dossier temporaire pour les images
└── ...
```

## Installation et configuration

1. **Installer les dépendances Python** :
   ```bash
   cd web_app
   pip install -r requirements.txt
   ```

2. **Ajouter votre modèle entraîné** :
   - Placez vos fichiers de modèle dans le dossier `models/`
   - Modifiez la fonction `predict_fibrosis()` dans `app.py` pour charger votre modèle

3. **Lancer l'application** :
   ```bash
   python app.py
   ```

4. **Accéder à l'interface** :
   Ouvrez votre navigateur à l'adresse : http://localhost:5000

## Intégration de votre modèle

### Étape 1 : Remplacer la fonction de prédiction

Dans `app.py`, remplacez la fonction `predict_fibrosis()` par votre modèle réel :

```python
def predict_fibrosis(image_array):
    # Charger votre modèle
    import tensorflow as tf  # ou pytorch
    
    # Exemple avec TensorFlow/Keras
    model = tf.keras.models.load_model('models/votre_modele.h5')
    
    # Préprocessing si nécessaire
    input_data = np.expand_dims(image_array, axis=0)
    
    # Prédiction
    predictions = model.predict(input_data)
    
    # Traiter les résultats selon votre architecture
    # ...
    
    return {
        'multiclass': {
            'predicted_class': predicted_class,
            'probabilities': probabilities_dict
        },
        'binary': binary_predictions_list
    }
```

### Étape 2 : Ajuster la configuration

Modifiez `MODELS_CONFIG` dans `app.py` si nécessaire :

```python
MODELS_CONFIG = {
    'fibrosis_hepatica': {
        'name': 'Detección de Fibrosis Hepática',
        'description': 'Votre description',
        'input_size': (256, 192),  # Ajustez selon vos besoins
        'original_size': (640, 480),
        # ...
    }
}
```

## Ajouter un nouveau modèle

Pour ajouter un nouveau modèle (par exemple, détection de tumeurs), ajoutez simplement une nouvelle entrée dans `MODELS_CONFIG` :

```python
MODELS_CONFIG = {
    'fibrosis_hepatica': {
        # Configuration existante...
    },
    'deteccion_tumores': {
        'name': 'Detección de Tumores',
        'description': 'Análisis automático para detección de tumores',
        'input_size': (224, 224),
        'original_size': (512, 512),
        'categories': {
            'Benigno': 'Tumor benigno',
            'Maligno': 'Tumor maligno',
            'Normal': 'Tejido normal'
        }
    }
}
```

Et créez une fonction de prédiction correspondante :

```python
def predict_tumores(image_array):
    # Votre logique de prédiction pour les tumeurs
    pass
```

## Formats d'images supportés

- PNG
- JPG/JPEG  
- TIFF
- BMP
- Taille maximale : 16MB

## Fonctionnalités de l'interface

### Page d'accueil
- Vue d'ensemble des modèles disponibles
- Informations techniques de chaque modèle
- Navigation intuitive

### Interface du modèle
- Upload d'image avec prévisualisation
- Traitement automatique aux bonnes dimensions
- Affichage des résultats :
  - Classification principale (F0-F4)
  - Classifications binaires (F0 vs F1-F4, F0-F1 vs F2-F4)
  - Probabilités pour chaque classe
  - Image traitée

## Déploiement en production

Pour déployer en production, utilisez un serveur WSGI comme Gunicorn :

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Sécurité

- Validation des types de fichiers
- Limitation de la taille des fichiers (16MB)
- Nettoyage automatique des fichiers temporaires
- Validation des entrées utilisateur

## Personnalisation

- **Couleurs** : Modifiez les variables CSS dans `static/css/style.css`
- **Textes** : Tous les textes sont en espagnol et facilement modifiables dans les templates
- **Layout** : Interface Bootstrap responsive, facilement personnalisable

## Support

Pour toute question ou problème :
1. Vérifiez que toutes les dépendances sont installées
2. Assurez-vous que les dimensions d'images sont correctes
3. Consultez les logs de la console pour les erreurs détaillées
