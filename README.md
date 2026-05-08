# MLOps Signal Pipeline

A minimal MLOps-style batch job that computes a rolling-mean binary signal on OHLCV market data. Demonstrates reproducibility, observability, and Docker-based deployment.

---

## Project structure

```
mlops-task/
├── run.py           # Main pipeline script
├── config.yaml      # Configuration (seed, window, version)
├── data.csv         # OHLCV input dataset (10 000 rows)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md        # This file
```

---

## Local run

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

The final metrics JSON is printed to **stdout**; detailed logs are written to **run.log**.

---

## Docker build & run

```bash
# Build the image
docker build -t mlops-task .

# Run (prints metrics JSON to stdout, exits 0 on success)
docker run --rm mlops-task
```

Outputs (`metrics.json` and `run.log`) live inside the container at `/app/`.  
To copy them to your host after the run:

```bash
# Run with a named container, then copy
docker run --name mlops-run mlops-task
docker cp mlops-run:/app/metrics.json ./metrics.json
docker cp mlops-run:/app/run.log     ./run.log
docker rm mlops-run
```

---

## Configuration (`config.yaml`)

| Key       | Type   | Description                                      |
|-----------|--------|--------------------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility            |
| `window`  | int    | Rolling-mean window size (rows)                  |
| `version` | string | Pipeline version tag written to `metrics.json`   |

```yaml
seed: 42
window: 5
version: "v1"
```

---

## Signal logic

1. Compute a rolling mean of `close` with the configured `window`.
2. The first `window - 1` rows produce `NaN` and are **excluded** from signal computation and the `rows_processed` count.
3. For every remaining row:
   - `signal = 1` if `close > rolling_mean`
   - `signal = 0` otherwise

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 21,
  "seed": 42,
  "status": "success"
}
```

> **rows_processed** is 9 996 (= 10 000 − 4 warm-up rows for window = 5).

### Error output

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Columns present: [...]"
}
```

---

## Validated error cases

| Scenario                        | Behaviour                                |
|---------------------------------|------------------------------------------|
| Input file missing              | Error metrics written, exit code 1       |
| Invalid / unparseable CSV       | Error metrics written, exit code 1       |
| Empty file                      | Error metrics written, exit code 1       |
| Missing `close` column          | Error metrics written, exit code 1       |
| Config file missing             | Error metrics written, exit code 1       |
| Missing config keys             | Error metrics written, exit code 1       |
| Invalid config value types      | Error metrics written, exit code 1       |

---

## Reproducibility

All runs with the same `config.yaml` and `data.csv` produce **identical** output because:
- `numpy.random.seed(seed)` is called before any computation.
- The rolling-mean and signal logic are fully deterministic.
