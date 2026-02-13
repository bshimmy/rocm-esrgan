FROM rocm-esrgan-base

WORKDIR /app

COPY upscale_video_once.py /app/
COPY watcher.py /app/

CMD ["/opt/venv/bin/python", "/app/watcher.py"]
