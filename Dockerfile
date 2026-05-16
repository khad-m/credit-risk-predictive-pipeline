# Use a lightweight Python environment
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Expose the API port
EXPOSE 8000

# Command to boot the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]