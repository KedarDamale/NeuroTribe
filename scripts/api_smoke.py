"""``python scripts/api_smoke.py`` - exercise every API surface end to end.

Verifies that each endpoint responds, that the binary surface buffers have the
exact byte counts the fsaverage5 geometry implies, and that the research-only
headers are present on every response.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000/api"

GET_ENDPOINTS = [
    "/health",
    "/dashboard",
    "/pipeline",
    "/blockers",
    "/system",
    "/system/config",
    "/system/tribe",
    "/system/preprocessing",
    "/data/sources",
    "/data/assets",
    "/data/summary",
    "/data/scans",
    "/stimulus",
    "/cohort?tier=PRIMARY",
    "/subjects",
    "/groups/runs",
    "/groups/results?tier=PRIMARY",
    "/qc",
    "/jobs",
    "/logs",
    "/logs/audit",
    "/reports",
    "/reports/provenance?tier=PRIMARY",
    "/surface/manifest",
    "/surface/parcellation",
]


def fetch(url: str, timeout: int = 180):
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.headers, response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    args = parser.parse_args()

    failures: list[str] = []
    print(f"NeuroTRIBE API smoke test against {args.base}\n")

    for endpoint in GET_ENDPOINTS:
        url = f"{args.base}{endpoint}"
        try:
            status, headers, body = fetch(url)
        except urllib.error.HTTPError as error:
            failures.append(f"{endpoint} -> HTTP {error.code}: {error.read()[:200]!r}")
            print(f"  FAIL  {endpoint:<42} HTTP {error.code}")
            continue
        except Exception as error:  # noqa: BLE001
            failures.append(f"{endpoint} -> {error}")
            print(f"  FAIL  {endpoint:<42} {error}")
            continue

        if headers.get("X-Research-Use-Only") != "true":
            failures.append(f"{endpoint} is missing the research-use-only header")

        try:
            payload = json.loads(body)
            shape = (f"{len(payload)} keys" if isinstance(payload, dict)
                     else f"{len(payload)} items")
        except json.JSONDecodeError:
            shape = f"{len(body)} bytes"
        print(f"  ok    {endpoint:<42} {status}  {shape}")

    # ---- surface geometry: exact byte counts -------------------------
    print("\nSurface geometry")
    try:
        _status, _headers, body = fetch(f"{args.base}/surface/manifest")
        manifest = json.loads(body)
        for hemi, info in manifest["hemispheres"].items():
            n_vertices = info["n_vertices"]
            n_faces = info["n_faces"]
            checks = [
                ("positions", n_vertices * 3 * 4),
                ("normals", n_vertices * 3 * 4),
                ("indices", n_faces * 3 * 4),
            ]
            for buffer_name, expected in checks:
                _s, _h, blob = fetch(f"{args.base}/surface/{hemi}/{buffer_name}")
                mark = "ok   " if len(blob) == expected else "FAIL "
                if len(blob) != expected:
                    failures.append(
                        f"surface/{hemi}/{buffer_name}: {len(blob)} bytes, expected {expected}"
                    )
                print(f"  {mark} {hemi}/{buffer_name:<10} {len(blob):>10,} bytes "
                      f"(expected {expected:,})")

            # First position must be a finite float32 triple.
            _s, _h, blob = fetch(f"{args.base}/surface/{hemi}/positions")
            x, y, z = struct.unpack_from("<3f", blob, 0)
            finite = all(abs(v) < 1e6 for v in (x, y, z))
            print(f"  {'ok   ' if finite else 'FAIL '} {hemi} first vertex "
                  f"({x:.2f}, {y:.2f}, {z:.2f})")
            if not finite:
                failures.append(f"{hemi} positions are not finite")

        total = sum(i["n_vertices"] for i in manifest["hemispheres"].values())
        expected_total = manifest["total_vertices"]
        print(f"  {'ok   ' if total == expected_total else 'FAIL '} "
              f"total vertices {total:,} (manifest says {expected_total:,})")
        if total != expected_total:
            failures.append("hemisphere vertex counts do not sum to total_vertices")

        _s, _h, labels = fetch(f"{args.base}/surface/parcellation/labels")
        expected_bytes = expected_total * 4
        mark = "ok   " if len(labels) == expected_bytes else "FAIL "
        print(f"  {mark} parcellation labels {len(labels):,} bytes "
              f"(expected {expected_bytes:,})")
        if len(labels) != expected_bytes:
            failures.append("parcellation label buffer has the wrong length")

    except Exception as error:  # noqa: BLE001
        failures.append(f"surface geometry: {error}")
        print(f"  FAIL  {error}")

    # ---- mutating endpoints ------------------------------------------
    print("\nActions")
    for endpoint in ("/pipeline/tick", "/data/rescan", "/stimulus/rescan"):
        url = f"{args.base}{endpoint}"
        try:
            request = urllib.request.Request(url, method="POST", data=b"")
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read())
            print(f"  ok    POST {endpoint:<37} {list(payload)[:4]}")
        except Exception as error:  # noqa: BLE001
            failures.append(f"POST {endpoint} -> {error}")
            print(f"  FAIL  POST {endpoint:<37} {error}")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"All {len(GET_ENDPOINTS)} endpoints + geometry + actions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
