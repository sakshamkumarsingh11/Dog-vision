# Python image use karein
FROM python:3.10-slim

# System dependencies for image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Requirements install karein
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Saara code copy karein
COPY . .

# Hugging Face port 7860 use karta hai
EXPOSE 7860

# Gunicorn use karein stable deployment ke liye
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]