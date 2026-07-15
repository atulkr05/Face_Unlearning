import json
import shutil
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

IMG_DIR = Path("/DATA2/Atul/2027/challenge/CelebAHQ/Img/img_celeba")
IDENTITY_FILE = Path("/DATA2/Atul/2027/challenge/CelebAHQ/Anno/identity_CelebA.txt")
SPLITS_FILE = Path("/DATA2/Atul/2027/challenge/face_unlearning/validation-splits.json")

OUTPUT_DIR = Path("/DATA2/Atul/2027/challenge/organized_validation_images")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Read identity mapping
# identity_CelebA.txt format:
# image_name identity_id
# -------------------------------------------------------

identity_to_images = defaultdict(list)

with open(IDENTITY_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        img_name, identity = line.split()
        identity_to_images[identity].append(img_name)

print(f"Loaded {len(identity_to_images)} identities.")

# -------------------------------------------------------
# Read validation splits
# -------------------------------------------------------

with open(SPLITS_FILE, "r") as f:
    splits = json.load(f)

# -------------------------------------------------------
# Copy images
# -------------------------------------------------------

for split in splits["splits"]:

    if split["track"] != "face":
        continue

    split_name = split["set"].replace(" ", "_")

    forget_id = str(split["forget_id"])
    retain_ids = [str(x) for x in split["retain_ids"]]

    print(f"\nProcessing {split_name}")

    #######################################################
    # Forget identity
    #######################################################

    forget_dir = OUTPUT_DIR / split_name / "forget" / forget_id
    forget_dir.mkdir(parents=True, exist_ok=True)

    for img_name in identity_to_images[forget_id]:
        src = IMG_DIR / img_name
        dst = forget_dir / img_name

        if src.exists():
            shutil.copy2(src, dst)

    #######################################################
    # Retain identities
    #######################################################

    for rid in retain_ids:

        retain_dir = OUTPUT_DIR / split_name / "retain" / rid
        retain_dir.mkdir(parents=True, exist_ok=True)

        for img_name in identity_to_images[rid]:

            src = IMG_DIR / img_name
            dst = retain_dir / img_name

            if src.exists():
                shutil.copy2(src, dst)

print("\nDone!")
print(f"Saved to: {OUTPUT_DIR}")