# Temel imaj olarak resmi Python 3.11-slim kullanıyoruz.
FROM python:3.11-slim

# Çalışma dizinini belirleyin.
WORKDIR /app

# Gereksinim dosyasını kopyalayın ve bağımlılıkları yükleyin.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyalayın.
COPY . .

# Uygulamayı çalıştırın.
CMD ["python", "main.py"]