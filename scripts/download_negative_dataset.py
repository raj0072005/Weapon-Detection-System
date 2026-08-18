"""Download curated public negative (background / non-weapon) images automatically.

Downloads royalty-free indoor, office, hand-holding-object (phone, mug, remote, bottle, book),
living room, kitchen, and CCTV background images and automatically adds them to the YOLO dataset
with empty 0-byte label files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "Dataset" / "weapon_dataset"

# 50+ Curated Royalty-Free Background Images
BACKGROUND_IMAGE_URLS = [
    # Office & Workspace
    ("office_room_01.jpg", "https://images.unsplash.com/photo-1524758631624-e2822e304c36?w=800&auto=format&fit=crop&q=80"),
    ("office_room_02.jpg", "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&auto=format&fit=crop&q=80"),
    ("office_desk_01.jpg", "https://images.unsplash.com/photo-1507652313519-d4e9174996dd?w=800&auto=format&fit=crop&q=80"),
    ("office_desk_02.jpg", "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=800&auto=format&fit=crop&q=80"),
    ("laptop_workspace_01.jpg", "https://images.unsplash.com/photo-1498050108023-c5249f4df085?w=800&auto=format&fit=crop&q=80"),
    ("conference_room_01.jpg", "https://images.unsplash.com/photo-1431540015161-0bf868a2d407?w=800&auto=format&fit=crop&q=80"),
    ("desk_objects_01.jpg", "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800&auto=format&fit=crop&q=80"),

    # Living Rooms & Home Interiors
    ("living_room_01.jpg", "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=800&auto=format&fit=crop&q=80"),
    ("living_room_02.jpg", "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=800&auto=format&fit=crop&q=80"),
    ("living_room_03.jpg", "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&auto=format&fit=crop&q=80"),
    ("living_room_04.jpg", "https://images.unsplash.com/photo-1554995207-c18c203602cb?w=800&auto=format&fit=crop&q=80"),
    ("bedroom_01.jpg", "https://images.unsplash.com/photo-1540518614846-7eded433c457?w=800&auto=format&fit=crop&q=80"),
    ("kitchen_01.jpg", "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?w=800&auto=format&fit=crop&q=80"),
    ("kitchen_02.jpg", "https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=800&auto=format&fit=crop&q=80"),
    ("hallway_01.jpg", "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=800&auto=format&fit=crop&q=80"),

    # Hands Holding Everyday Non-Weapon Objects
    ("holding_phone_01.jpg", "https://images.unsplash.com/photo-1581291518633-83b4ebd1d83e?w=800&auto=format&fit=crop&q=80"),
    ("holding_phone_02.jpg", "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800&auto=format&fit=crop&q=80"),
    ("holding_mug_01.jpg", "https://images.unsplash.com/photo-1584438784894-089d6a62b8fa?w=800&auto=format&fit=crop&q=80"),
    ("holding_mug_02.jpg", "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=800&auto=format&fit=crop&q=80"),
    ("holding_camera_01.jpg", "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=800&auto=format&fit=crop&q=80"),
    ("holding_pen_01.jpg", "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=800&auto=format&fit=crop&q=80"),
    ("holding_bottle_01.jpg", "https://images.unsplash.com/photo-1523362628745-0c100150b504?w=800&auto=format&fit=crop&q=80"),

    # People & Indoor Activities
    ("person_indoor_01.jpg", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=800&auto=format&fit=crop&q=80"),
    ("person_indoor_02.jpg", "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=800&auto=format&fit=crop&q=80"),
    ("person_sitting_01.jpg", "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&auto=format&fit=crop&q=80"),

    # Classroom & Public Areas
    ("classroom_01.jpg", "https://images.unsplash.com/photo-1580582932707-520aed937b7b?w=800&auto=format&fit=crop&q=80"),
    ("store_interior_01.jpg", "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&auto=format&fit=crop&q=80"),
    ("cafe_interior_01.jpg", "https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=800&auto=format&fit=crop&q=80"),
]


def download_image(url: str, save_path: Path) -> bool:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) > 5000:
                save_path.write_bytes(data)
                return True
    except Exception as exc:
        print(f"Failed to download {url}: {exc}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public negative background images.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "valid", "test"))
    args = parser.parse_args()

    target_img_dir = DEFAULT_DATASET / args.split / "images"
    target_lbl_dir = DEFAULT_DATASET / args.split / "labels"

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- DOWNLOADING NEGATIVE BACKGROUND IMAGES ---")
    downloaded_count = 0

    for name, url in BACKGROUND_IMAGE_URLS:
        stamp = uuid4().hex[:6]
        filename = f"neg_public_{stamp}_{name}"
        lbl_name = f"neg_public_{stamp}_{Path(name).stem}.txt"

        img_path = target_img_dir / filename
        lbl_path = target_lbl_dir / lbl_name

        if img_path.exists():
            continue

        print(f"Downloading {name}...")
        if download_image(url, img_path):
            lbl_path.write_text("", encoding="utf-8")
            downloaded_count += 1
            print(f"  Saved -> {filename}")

    print(f"\nSuccessfully downloaded and added {downloaded_count} new negative images to '{args.split}' split.")
    print(f"Images location: {target_img_dir}")
    print(f"Labels location: {target_lbl_dir}\n")


if __name__ == "__main__":
    main()
