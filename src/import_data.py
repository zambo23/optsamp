"""
Download the source datasets into ./data/
──────────────────────────────────────────
credit_risk : OpenML 43454 "Credit-Risk-Dataset" (32 581 loans, target
              loan_status)  →  data/credit_risk.csv
german      : German Credit, raw UCI file (1 000 rows, no header)
              →  data/german.csv

Usage
─────
  python src/import_data.py                     # credit_risk (default)
  python src/import_data.py --dataset german
  python src/import_data.py --dataset all --force   # re-download everything
"""

import argparse
import os
import ssl
import sys
import urllib.request
from pathlib import Path

GERMAN_URL     = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
OPENML_ID      = 43454
DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
GERMAN_FILE    = DATA_DIR / "german.csv"
CREDIT_RISK_FILE = DATA_DIR / "credit_risk.csv"


def ssl_context() -> ssl.SSLContext:
    """
    Use certifi's CA bundle when available: python.org builds on macOS
    ship without system certificates, so the default context fails.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        print(f"   {dest} already exists (use --force to overwrite)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"   Downloading {url} …", end=" ", flush=True)
    with urllib.request.urlopen(url, context=ssl_context(), timeout=30) as resp:
        payload = resp.read()

    # write to a temp file first so a failed download never leaves a partial csv
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)

    n_rows = payload.decode("utf-8").strip().count("\n") + 1
    print(f"done  ({n_rows} rows, {len(payload):,} bytes)")
    return dest


def fetch_credit_risk(dest: Path = CREDIT_RISK_FILE, force: bool = False) -> Path:
    """Fetch OpenML 43454 with scikit-learn and save it as a headed csv."""
    if dest.exists() and not force:
        print(f"   {dest} already exists (use --force to overwrite)")
        return dest

    # sklearn opens its own connections, so point OpenSSL at certifi's bundle
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    from sklearn.datasets import fetch_openml

    print(f"   Fetching OpenML dataset {OPENML_ID} …", end=" ", flush=True)
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True,
                         data_home=DATA_DIR / "openml_cache").frame

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(dest)
    print(f"done  ({len(frame)} rows × {frame.shape[1]} cols)")
    return dest


DATASETS = {
    "credit_risk": lambda force: fetch_credit_risk(CREDIT_RISK_FILE, force),
    "german"     : lambda force: download(GERMAN_URL, GERMAN_FILE, force),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset", choices=[*DATASETS, "all"],
                        default="credit_risk")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the file already exists")
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        try:
            path = DATASETS[name](args.force)
        except Exception as exc:
            print(f"\n   ✗ {name}: download failed: {exc}", file=sys.stderr)
            return 1
        print(f"✓ Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
