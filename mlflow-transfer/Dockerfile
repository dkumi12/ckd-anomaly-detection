FROM ghcr.io/mlflow/mlflow:latest

RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir psycopg2-binary

RUN printf 'events {}\nhttp {\n  server {\n    listen 8080;\n    location / {\n      proxy_pass http://127.0.0.1:5000;\n      proxy_set_header Host localhost;\n    }\n  }\n}\n' > /etc/nginx/nginx.conf

RUN printf '#!/bin/bash\nset -e\nmlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" --default-artifact-root gs://chronic-kd-mlflow-artifacts/ &\nsleep 5\nnginx -g "daemon off;"\n' > /start.sh && chmod +x /start.sh

EXPOSE 8080
ENTRYPOINT []
CMD ["/bin/bash", "/start.sh"]
