"""T042 — OpenAPI freshness gate.

Ensures the live FastAPI schema matches the checked-in contract file. When
this test fails, the developer changed the API shape and must regenerate
the contract:

    uv run dump-openapi

…then commit the updated `specs/001-multi-model-asr/contracts/openapi.json`
as part of the same PR. Constitution II makes this file the source of
truth for the API contract.
"""

import json
from pathlib import Path

from asr.app import create_app

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "001-multi-model-asr"
    / "contracts"
    / "openapi.json"
)


def test_openapi_schema_matches_checked_in_contract():
    app = create_app(models=[])
    live = app.openapi()
    on_disk = json.loads(CONTRACT_PATH.read_text())

    assert live == on_disk, (
        "Live OpenAPI schema differs from checked-in contract. "
        "Run `uv run dump-openapi` and commit "
        f"{CONTRACT_PATH.relative_to(Path.cwd())}."
    )
