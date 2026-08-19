"""Download and validate the configured pretrained TRIBE v2 model.

Run by the one-shot ``tribe-bootstrap`` Compose service.  Package installation
happens at image build time; this process materialises model weights in the
shared persistent cache volume before the API and worker are allowed to start.
"""

from __future__ import annotations

import json
import sys

from neurotribe.config import get_settings
from neurotribe.tribe.model import TribeUnavailable, load


def main() -> int:
    settings = get_settings()
    try:
        loaded = load(settings)
    except TribeUnavailable as exc:
        print(f"TRIBE v2 preload failed: {exc}", file=sys.stderr)
        return 1

    if loaded.is_mock:
        print("TRIBE v2 preload resolved to the mock backend; refusing to continue.", file=sys.stderr)
        return 1

    print(json.dumps({"status": "ready", **loaded.manifest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
