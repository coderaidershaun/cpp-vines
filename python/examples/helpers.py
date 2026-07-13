from csv import DictReader
from pathlib import Path

from vines import Asset


ASSET_NAMES = ("BTC", "ETH", "LINK", "XRP", "SOL", "DOGE")
PRICES_PATH = Path(__file__).resolve().parents[2] / "prices.csv"


def load_assets(prices_path: str | Path = PRICES_PATH) -> list[Asset]:
    path = Path(prices_path)

    with path.open(encoding="utf-8-sig", newline="") as prices_file:
        reader = DictReader(prices_file)
        missing_columns = set(ASSET_NAMES).difference(reader.fieldnames or ())

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing asset price columns: {missing}")

        rows = list(reader)

    if not rows:
        raise ValueError("asset price data is empty")

    assets = [Asset(len(rows)) for _ in ASSET_NAMES]

    for row in rows:
        for name, asset in zip(ASSET_NAMES, assets, strict=True):
            asset.push_price(float(row[name]))

    return assets
