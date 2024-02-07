FROM python:3.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY src/backend/ ./

RUN python manage.py collectstatic --no-input

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "facedetect.wsgi:application"]