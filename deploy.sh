#!/bin/bash

# Script de déploiement pour serveur de production
echo "🚀 Déploiement de l'application de diagnostic IA"
echo "================================================="

# Variables d'environnement
export FLASK_ENV=production
export SECRET_KEY=$(openssl rand -hex 16)

# Créer les dossiers nécessaires
echo "📁 Création des dossiers nécessaires..."
mkdir -p uploads
mkdir -p models
mkdir -p static/css
mkdir -p templates

# Installer les dépendances
echo "📦 Installation des dépendances..."
pip3 install -r requirements.txt

# Donner les permissions nécessaires
echo "🔒 Configuration des permissions..."
chmod 755 uploads
chmod 755 models

# Lancer l'application avec Gunicorn
echo "🌐 Lancement de l'application sur le port 5000..."
echo "L'application sera accessible via: http://150.239.171.57:5000"
echo "================================================="

gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
