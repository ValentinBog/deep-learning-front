# 🚀 Guide de Déploiement - Application de Diagnostic IA

## 📋 Instructions de Déploiement sur le Serveur Cloud

### Informations du Serveur
- **IP**: 150.239.171.57
- **Utilisateur**: Administrator  
- **Mot de passe**: DV5o1Ew7EeAlHRdt4Xyt
- **Accès**: Bureau à distance (Remote Desktop)

### 🔧 Étapes de Déploiement

#### 1. Préparation Locale
```bash
# Empaqueter l'application
./package.sh
```

#### 2. Connexion au Serveur
1. Ouvrir un client Bureau à distance
2. Se connecter à: `150.239.171.57`
3. Utiliser les identifiants fournis

#### 3. Transfert des Fichiers
1. Copier le fichier `.tar.gz` généré vers le serveur
2. Utiliser le presse-papier du bureau à distance ou un transfert de fichiers

#### 4. Installation sur le Serveur
```bash
# Extraire l'archive
tar -xzf deep-learning-app-YYYYMMDD-HHMMSS.tar.gz

# Entrer dans le dossier
cd deep-learning-front

# Lancer le déploiement
./deploy.sh
```

#### 5. Accès à l'Application
Une fois déployée, l'application sera accessible via:
```
http://150.239.171.57:5000
```

### 📁 Structure du Projet
```
deep-learning-front/
├── app.py                          # Application Flask principale
├── best_unetpp_vgg16_multitask.pth # Modèle de deep learning
├── model_wrapper.py                # Interface du modèle
├── config.py                       # Configuration
├── requirements.txt                # Dépendances Python
├── deploy.sh                       # Script de déploiement
├── static/                         # Fichiers statiques (CSS, JS)
├── templates/                      # Templates HTML
└── uploads/                        # Dossier pour les images uploadées
```

### 🔧 Dépannage

#### Port déjà utilisé
Si le port 5000 est occupé, modifier dans `deploy.sh`:
```bash
gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 120 app:app
```

#### Problèmes de permissions
```bash
sudo chown -R $USER:$USER /path/to/app
chmod -R 755 uploads/
```

#### Redémarrer l'application
```bash
pkill -f gunicorn
./deploy.sh
```

### 🧪 Tests de Fonctionnement

1. **Test de l'interface**: Accéder à l'URL principale
2. **Test d'upload**: Uploader une image de test
3. **Test du modèle**: Vérifier que le modèle process les images
4. **Test de suppression**: Tester la suppression des dossiers/images

### 📞 Support
En cas de problème, vérifier:
- Les logs dans le terminal où `deploy.sh` est lancé
- La présence du fichier modèle `best_unetpp_vgg16_multitask.pth`
- Les permissions des dossiers `uploads/` et `models/`
