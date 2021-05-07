FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Copy model files
COPY ./models /app/models


# Copy requirements.txt
COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy rest
COPY ./ /app


# CMD [ "python", "./your-daemon-or-script.py" ]