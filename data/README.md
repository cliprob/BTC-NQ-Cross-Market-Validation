# Data contract

Raw market data is intentionally excluded from this public repository.

## BTCUSDT perpetual futures

Expected columns (case-insensitive):

```text
open time, open, high, low, close, volume
```

`open time` must be an ISO timestamp or Unix timestamp representing the bar open in UTC.

## Nasdaq front-month futures

Expected columns:

```text
date, open, high, low, close, volume, contract
```

`date` uses `YYYYMMDD HH:MM:SS` in `America/Chicago`. `contract` identifies the actual quarterly
contract (for example `202603`). Events whose feature or label window crosses a contract change are
excluded.

Copy `configs/research.yaml` to `configs/research.local.yaml`, point it at local files, and run the
validation stage before the final stage. The committed test suite generates a small synthetic fixture,
so CI never needs licensed data.

