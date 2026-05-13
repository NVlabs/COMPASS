#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Search SAGE-10k dataset for scenes matching a text query.

Uses the HuggingFace API to list scenes, downloads layout JSONs on demand,
and caches them locally for fast repeated searches.

Usage:
    python scripts/sage10k_search.py "bedroom with desk"
    python scripts/sage10k_search.py "warehouse" --top 10
    python scripts/sage10k_search.py --list-room-types
    python scripts/sage10k_search.py "kitchen" --download --output-dir ./sage_scenes
"""

import argparse
import json
import os
import random
import re
import sys
import zipfile

from huggingface_hub import HfApi, hf_hub_download

DATASET_ID = "nvidia/SAGE-10k"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "sage_10k_cache")


def get_api():
    return HfApi()


def list_scene_files(api, cache_dir):
    """List all scene zip files in the dataset, with caching."""
    index_path = os.path.join(cache_dir, "scene_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)

    print("Building scene index from HuggingFace API (first time only)...", file=sys.stderr)
    files = []
    for item in api.list_repo_tree(DATASET_ID, path_in_repo="scenes", repo_type="dataset"):
        name = getattr(item, "rfilename", None) or getattr(item, "path", "")
        if name.endswith(".zip"):
            files.append(name)

    os.makedirs(cache_dir, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(files, f)
    print(f"Indexed {len(files)} scenes.", file=sys.stderr)
    return files


def extract_layout_id(zip_filename):
    """Extract layout ID from zip filename like '20251213_020526_layout_84b703fb.zip'."""
    match = re.search(r"layout_([a-f0-9]+)", zip_filename)
    return match.group(1) if match else os.path.splitext(os.path.basename(zip_filename))[0]


def download_layout_json(_api, zip_path, cache_dir):
    """Download a scene zip and extract its layout JSON."""
    layout_id = extract_layout_id(zip_path)
    json_cache = os.path.join(cache_dir, "layouts", f"{layout_id}.json")

    if os.path.exists(json_cache):
        with open(json_cache) as f:
            return json.load(f)

    local_zip = hf_hub_download(
        repo_id=DATASET_ID,
        filename=zip_path,
        repo_type="dataset",
        local_dir=os.path.join(cache_dir, "zips"),
    )

    os.makedirs(os.path.join(cache_dir, "layouts"), exist_ok=True)
    with zipfile.ZipFile(local_zip) as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json") and "layout" in n.lower()]
        if not json_files:
            json_files = [n for n in zf.namelist() if n.endswith(".json")]
        if json_files:
            data = json.loads(zf.read(json_files[0]))
            with open(json_cache, "w") as f:
                json.dump(data, f)
            return data
    return None


def extract_scene_metadata(layout_data):
    """Extract searchable metadata from a layout JSON."""
    meta = {
        "id": layout_data.get("id", "unknown"),
        "description": layout_data.get("description", ""),
        "building_style": layout_data.get("building_style", ""),
        "room_types": [],
        "object_count": 0,
        "rooms": [],
    }
    for room in layout_data.get("rooms", []):
        room_type = room.get("room_type", "unknown")
        objects = room.get("objects", [])
        meta["room_types"].append(room_type)
        meta["object_count"] += len(objects)
        meta["rooms"].append({
            "room_type": room_type,
            "object_count": len(objects),
        })
    return meta


def score_match(query, meta):
    """Score how well a scene matches the query. Higher = better match."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    score = 0

    for rt in meta["room_types"]:
        rt_lower = rt.lower()
        if query_lower in rt_lower or rt_lower in query_lower:
            score += 10
        for word in query_words:
            if word in rt_lower:
                score += 5

    desc_lower = meta["description"].lower()
    for word in query_words:
        if word in desc_lower:
            score += 3

    style_lower = meta["building_style"].lower()
    for word in query_words:
        if word in style_lower:
            score += 2

    return score


def search_scenes(query, scene_files, api, cache_dir, top_n=5, sample_size=100):
    """Search scenes by downloading and checking layout JSONs.

    To avoid downloading all 10K scenes, we first filter by filename patterns,
    then sample and download layout JSONs for detailed matching.
    """
    query_lower = query.lower()

    # Phase 1: Score by filename (free, no downloads)
    filename_scored = []
    for f in scene_files:
        fname_lower = os.path.basename(f).lower()
        fscore = 0
        for word in query_lower.split():
            if word in fname_lower:
                fscore += 5
        filename_scored.append((f, fscore))

    # Phase 2: Download layout JSONs for a sample of scenes
    # Prioritize filename matches, then take random sample
    filename_scored.sort(key=lambda x: -x[1])
    candidates = [f for f, s in filename_scored[:sample_size]]

    results = []
    for i, zip_path in enumerate(candidates):
        print(f"\rSearching... ({i+1}/{len(candidates)})", end="", file=sys.stderr)
        try:
            layout = download_layout_json(api, zip_path, cache_dir)
            if layout is None:
                continue
            meta = extract_scene_metadata(layout)
            meta["zip_path"] = zip_path
            score = score_match(query, meta)
            if score > 0:
                results.append((score, meta))
        except Exception as e:    # pylint: disable=broad-exception-caught
            print(f"\nWarning: failed to process {zip_path}: {e}", file=sys.stderr)
            continue

    print("", file=sys.stderr)
    results.sort(key=lambda x: -x[0])
    return results[:top_n]


def list_room_types(scene_files, api, cache_dir, sample_size=50):
    """Sample scenes to discover available room types."""
    room_types = set()
    sample = random.sample(scene_files, min(sample_size, len(scene_files)))
    for i, zip_path in enumerate(sample):
        print(f"\rSampling room types... ({i+1}/{len(sample)})", end="", file=sys.stderr)
        try:
            layout = download_layout_json(api, zip_path, cache_dir)
            if layout:
                meta = extract_scene_metadata(layout)
                room_types.update(meta["room_types"])
        except Exception:    # pylint: disable=broad-exception-caught
            continue
    print("", file=sys.stderr)
    return sorted(room_types)


def download_scene(_api, zip_path, output_dir, cache_dir):
    """Download and extract a full scene."""
    local_zip = hf_hub_download(
        repo_id=DATASET_ID,
        filename=zip_path,
        repo_type="dataset",
        local_dir=os.path.join(cache_dir, "zips"),
    )

    layout_id = extract_layout_id(zip_path)
    scene_dir = os.path.join(output_dir, layout_id)
    os.makedirs(scene_dir, exist_ok=True)

    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(scene_dir)

    print(f"Scene extracted to: {scene_dir}")
    return scene_dir


def main():
    parser = argparse.ArgumentParser(description="Search SAGE-10k scenes")
    parser.add_argument("query", nargs="?", help="Search query (e.g. 'bedroom with desk')")
    parser.add_argument("--top", type=int, default=5, help="Number of results")
    parser.add_argument("--sample-size",
                        type=int,
                        default=100,
                        help="Number of scenes to sample for search")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Cache directory")
    parser.add_argument("--list-room-types", action="store_true", help="List discovered room types")
    parser.add_argument("--download", action="store_true", help="Download the top result")
    parser.add_argument("--output-dir",
                        default="./sage_10k_scenes",
                        help="Output directory for downloads")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    api = get_api()
    scene_files = list_scene_files(api, args.cache_dir)

    if args.list_room_types:
        types = list_room_types(scene_files, api, args.cache_dir)
        print("Discovered room types:")
        for t in types:
            print(f"  - {t}")
        return

    if not args.query:
        parser.error("Please provide a search query or use --list-room-types")

    results = search_scenes(args.query,
                            scene_files,
                            api,
                            args.cache_dir,
                            top_n=args.top,
                            sample_size=args.sample_size)

    if not results:
        print("No matching scenes found. Try a broader query or increase --sample-size.")
        return

    if args.json:
        print(json.dumps([{"score": s, **m} for s, m in results], indent=2))
    else:
        print(f"\nTop {len(results)} matches for '{args.query}':\n")
        for i, (_, meta) in enumerate(results, 1):
            room_types = ", ".join(meta["room_types"]) or "unknown"
            print(f"  {i}. {meta['id']}")
            print(f"     Room types: {room_types}")
            print(f"     Style: {meta['building_style'] or 'N/A'}")
            print(f"     Objects: {meta['object_count']}")
            print(f"     Description: {meta['description'][:100]}..." if len(
                meta.get('description', '')) >
                  100 else f"     Description: {meta['description'] or 'N/A'}")
            print(f"     File: {meta['zip_path']}")
            print()

    if args.download and results:
        _, top_meta = results[0]
        print(f"Downloading top result: {top_meta['id']}...")
        scene_dir = download_scene(api, top_meta["zip_path"], args.output_dir, args.cache_dir)
        print(f"Done. Scene at: {scene_dir}")


if __name__ == "__main__":
    main()
