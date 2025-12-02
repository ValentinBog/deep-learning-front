#!/bin/bash

# Script pour empaqueter l'application pour le déploiement
echo "📦 Préparation du package de déploiement..."

# Nom du fichier de sauvegarde
ARCHIVE_NAME="deep-learning-app-$(date +%Y%m%d-%H%M%S).tar.gz"

# Créer l'archive en excluant les fichiers inutiles
tar -czf "$ARCHIVE_NAME" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".git" \
    --exclude="venv" \
    --exclude="*.log" \
    --exclude="uploads/*" \
    .

echo "✅ Archive créée: $ARCHIVE_NAME"
echo "📁 Taille de l'archive: $(du -h "$ARCHIVE_NAME" | cut -f1)"
echo ""
echo "🚀 Instructions de déploiement:"
echo "1. Copiez cette archive sur le serveur: 150.239.171.57"
echo "2. Connectez-vous au serveur avec Bureau à distance"
echo "3. Extraire l'archive: tar -xzf $ARCHIVE_NAME"
echo "4. Entrer dans le dossier: cd deep-learning-front"
echo "5. Lancer le déploiement: ./deploy.sh"
