#!/usr/bin/env python3
"""Add optional RMS playback to a published gallery using saved raw WAVs only."""
import argparse
import importlib.util
import json
from pathlib import Path

from results_playback import add_rms_playback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Saved per-sample raw inference directory")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    arguments = parser.parse_args()
    spec = importlib.util.spec_from_file_location("gallery", Path(__file__).with_name("build-results-page.py"))
    gallery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gallery)
    manifest_path = arguments.docs / "results.json"
    manifest = json.loads(manifest_path.read_text())
    add_rms_playback(manifest, arguments.docs, arguments.source, gallery.spectrogram)
    gallery.write_json(manifest_path, manifest)
    print(json.dumps({"samples": len(manifest["samples"]), "checkpoint": manifest["checkpoint"]["step"],
                      "max_rms_spread_db": max(s["rms_matched"]["maximum_rms_spread_db"] for s in manifest["samples"])}))


if __name__ == "__main__":
    main()
