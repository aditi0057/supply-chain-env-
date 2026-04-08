# HuggingFace Spaces Docker configuration
FROM python:3.11-slim

# Create non-root user (HuggingFace requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project files
COPY --chown=user . /app

# HuggingFace requires port 7860
EXPOSE 7860

# Start the server
ENV PYTHONPATH="/app"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "/app"]