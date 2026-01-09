FROM python:3.11-slim

WORKDIR /app

# تثبيت المتطلبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ التطبيق
COPY app ./app

# Railway يحدد PORT
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main"]
