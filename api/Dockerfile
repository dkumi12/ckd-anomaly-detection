# Use Python 3.10 as mandated by your stack
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# libgomp1 is required by LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI script (main.py) into the container
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Start the API server, ensuring it listens on all network interfaces (0.0.0.0)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]