"""Dump the live FastAPI app's OpenAPI schema to the checked-in contract file.

Run via: `uv run dump-openapi`

The freshness contract test (`tests/contract/test_openapi_freshness.py`)
asserts that the result of `create_app().openapi()` deep-equals the
checked-in copy. When this test fails, run this script to regenerate the
file, review the diff in your PR, and commit the change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "specs"
    / "001-multi-model-asr"
    / "contracts"
    / "openapi.json"
)


def main() -> None:
    from asr.app import create_app

    app = create_app(models=[])
    schema = app.openapi()
    out = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    CONTRACT_PATH.write_text(out)
    sys.stderr.write(f"wrote {CONTRACT_PATH} ({len(out)} bytes)\n")


if __name__ == "__main__":
    main()
