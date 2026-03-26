# Use a lightweight Python base image
FROM python:3.12-slim

# Copy the pre-compiled uv binary from the official Astral image. 
# This is the fastest way to get uv inside Docker.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Hugging Face Mandate: Create a non-root user
RUN useradd -m -u 1000 user
USER user

# Set up environment variables
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/home/user/app/.venv

# Set the working directory
WORKDIR $HOME/app

# Step 1: Copy ONLY your dependency files first.
# This caches your installed packages so Docker doesn't re-download 
# everything every time you change a single line of code in app.py.
COPY --chown=user:user pyproject.toml uv.lock ./

# Step 2: Install dependencies natively using uv
# --frozen ensures it strictly follows your lockfile
# --no-dev skips installing testing tools you don't need in production
RUN uv sync --frozen --no-dev

# Step 3: Copy the rest of your actual code over
COPY --chown=user:user . .

# Expose the exact port Hugging Face looks for
EXPOSE 7860

# Command to run the app using uv's virtual environment
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]