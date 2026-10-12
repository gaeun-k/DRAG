FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-data.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-data.txt
COPY . .
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
