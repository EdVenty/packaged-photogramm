FROM python:3.12-slim
COPY tui /src/tui
RUN pip install --no-cache-dir /src/tui && rm -rf /src/tui
ENV PG_SERVER=http://api:8000 TERM=xterm-256color
WORKDIR /host
ENTRYPOINT ["pgtui"]
