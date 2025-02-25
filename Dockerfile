FROM python:3.13-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]