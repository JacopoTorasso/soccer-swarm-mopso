import csv
import io
import json


def format_backtest_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def format_backtest_json(summary: dict) -> str:
    return json.dumps(summary, indent=2, default=str)


def format_predictions_json(predictions: list[dict]) -> str:
    serializable = []
    for p in predictions:
        row = {}
        for k, v in p.items():
            if isinstance(v, tuple):
                row[k] = list(v)
            else:
                row[k] = v
        serializable.append(row)
    return json.dumps(serializable, indent=2, default=str)
