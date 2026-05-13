# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the necessary dependencies, including FastAPI, Uvicorn, and scikit-learn pinned to 1.6.1
# as it matches your training environment and model compatibility
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn pydantic "scikit-learn==1.6.1"

# Copy the script and the model into the container
COPY predict_by_model.py .
COPY obesity_xgb_top10.joblib .

# Expose the port that the FastAPI app runs on
EXPOSE 8008

# Command to run the server
CMD ["uvicorn", "predict_by_model:app", "--host", "0.0.0.0", "--port", "8008"]
