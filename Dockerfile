FROM python:3.11-slim

WORKDIR /research
COPY pyproject.toml requirements.txt ./
COPY cross_market ./cross_market
RUN pip install --no-cache-dir -e .
COPY configs ./configs
COPY data ./data
ENTRYPOINT ["python", "-m", "cross_market.run"]
