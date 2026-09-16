"""Download public Binance USD-M BTCUSDT one-minute klines."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"


def millis(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000)


def download(start: str, end: str, output: Path) -> None:
    cursor, stop = millis(start), millis(end)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Open time", "Open", "High", "Low", "Close", "Volume"])
        while cursor < stop:
            query = urllib.parse.urlencode(
                {"symbol": "BTCUSDT", "interval": "1m", "startTime": cursor, "endTime": stop, "limit": 1500}
            )
            with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=30) as response:  # noqa: S310
                rows = json.load(response)
            if not rows:
                break
            for row in rows:
                writer.writerow([datetime.fromtimestamp(row[0] / 1_000, UTC).isoformat(), *row[1:6]])
            cursor = int(rows[-1][0]) + 60_000
            time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    download(args.start, args.end, args.output)


if __name__ == "__main__":
    main()
