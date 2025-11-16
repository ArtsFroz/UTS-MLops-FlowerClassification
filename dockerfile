# Gunakan base image Python
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Install dependency OS yang diperlukan (opsional tapi aman)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements terlebih dahulu (biar layer cache efisien)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code dan model ke dalam container
COPY . .

# Expose port default Gradio
EXPOSE 7860

# Jalankan aplikasi
CMD ["python", "app.py"]
