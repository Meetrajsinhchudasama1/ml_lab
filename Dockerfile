# Use official Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src
COPY app.py .

# Run the demo script by default
CMD ["python", "src/demo.py"]

