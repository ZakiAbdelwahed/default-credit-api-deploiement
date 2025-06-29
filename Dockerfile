FROM python:3.10

WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Exposer le port
EXPOSE $PORT

# Commande de démarrage avec port par défaut
CMD ["sh", "-c", "uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000} --workers=1"]