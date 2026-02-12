# Base image 
FROM python:3.11-slim

# Set working dir
WORKDIR /app 

# copy and install dependencies 
COPY docker/inference/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 

# copy server to working dir 
COPY docker/inference/server.py . 

# Expose port 8081
EXPOSE 8081

# run server.py
CMD ["python", "server.py"]


# docker run --rm -p 8081:8080 \
#   -v wine_model_storage:/app/models \
#   -v "$(pwd)/logs:/app/logs" \
#   mlip-inference
