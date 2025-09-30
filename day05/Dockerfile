# Use official Python image
FROM python:3.13.2

# Set working directory inside the container
WORKDIR /app

# Copy dependencies file
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser
# Create a non-root user
RUN chown -R appuser:appuser /app
# Switch to the non-root user
USER appuser

# Copy all app files
COPY . /app

# Expose port 8000
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
