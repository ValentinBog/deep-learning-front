#!/bin/bash

# Script de lancement pour le système de diagnostic par IA

echo "🏥 Sistema de Diagnóstico por IA - Iniciando..."
echo "=============================================="

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Vérifier si pip est installé
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 n'est pas installé. Veuillez l'installer d'abord."
    exit 1
fi

# Créer un environnement virtuel si il n'existe pas
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🔧 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances
echo "📥 Installation des dépendances..."
pip install -r requirements.txt

# Créer les dossiers nécessaires
mkdir -p uploads
mkdir -p models

echo "✅ Configuration terminée!"
echo ""
echo "🚀 Lancement du serveur..."
echo "📱 L'application sera disponible sur: http://localhost:5000"
echo "⏹️  Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

# Lancer l'application
python app.py
