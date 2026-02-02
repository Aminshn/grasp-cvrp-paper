FROM python:3.9-slim-bullseye

USER root
RUN apt-get update && apt-get install -y curl gnupg2 unixodbc unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pyodbc sqlalchemy

COPY cvrp_streamlit/ /app/cvrp_streamlit/
COPY solver/ /app/solver/
COPY benchmarks/ /app/benchmarks/

ENV PYTHONPATH="${PYTHONPATH}:/app"

CMD ["python", "solver/worker.py"]