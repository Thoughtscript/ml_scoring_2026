FROM python:3.11-slim

WORKDIR /lab

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/lab PYTHONUNBUFFERED=1
CMD ["python", "run_all.py"]
