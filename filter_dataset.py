#!/usr/bin/env python3
"""
Filter billboard dataset using CLIP zero-shot classification.

Scores each image as "outdoor advertising billboard" vs unrelated categories,
then copies the top-scoring images to a clean dataset folder.
"""

import os
import shutil
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# Config
DATASET_DIR = Path("mega_dataset/train/images")
LABELS_DIR = Path("mega_dataset/train/labels")
OUTPUT_DIR = Path("clean_dataset")
SCORES_FILE = Path("clip_scores.json")

# How many to keep
TARGET_COUNT = 1200  # Aim for 1200, then manually trim to ~1000

# CLIP categories
BILLBOARD_PROMPTS = [
    "an outdoor advertising billboard on a road",
    "a large billboard with an advertisement",
    "a roadside billboard sign with a commercial ad",
    "a building-mounted advertising billboard",
    "a digital advertising billboard screen",
]

NOT_BILLBOARD_PROMPTS = [
    "a motorcycle or vehicle",
    "a road sign or traffic sign",
    "a train station or bus station sign",
    "a restaurant or shop storefront",
    "people or animals without any billboard",
    "a sports stadium or racetrack",
    "a van or bus with text on it",
    "an indoor scene",
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Encode text prompts
    billboard_texts = tokenizer(BILLBOARD_PROMPTS).to(device)
    not_billboard_texts = tokenizer(NOT_BILLBOARD_PROMPTS).to(device)

    with torch.no_grad():
        billboard_features = model.encode_text(billboard_texts)
        billboard_features /= billboard_features.norm(dim=-1, keepdim=True)

        not_billboard_features = model.encode_text(not_billboard_texts)
        not_billboard_features /= not_billboard_features.norm(dim=-1, keepdim=True)

    # Get all images
    images = sorted(DATASET_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images to score")

    # Skip genie source entirely (known garbage)
    images = [img for img in images if not img.name.startswith("billboard_genie_")]
    print(f"After removing genie source: {len(images)} images")

    # Score each image
    scores = {}
    batch_size = 32

    print(f"\nScoring images in batches of {batch_size}...")

    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(preprocess(img))
            except Exception as e:
                print(f"  Error loading {img_path.name}: {e}")
                batch_images.append(None)

        # Filter out failed loads
        valid_indices = [j for j, img in enumerate(batch_images) if img is not None]
        if not valid_indices:
            continue

        valid_images = torch.stack([batch_images[j] for j in valid_indices]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(valid_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Score: max similarity to billboard prompts - max similarity to not-billboard
            billboard_sim = (image_features @ billboard_features.T).max(dim=1).values
            not_billboard_sim = (image_features @ not_billboard_features.T).max(dim=1).values

            # Net score: how much more "billboard" than "not billboard"
            net_scores = (billboard_sim - not_billboard_sim).cpu().numpy()

        for idx, j in enumerate(valid_indices):
            img_name = batch_paths[j].name
            scores[img_name] = float(net_scores[idx])

        if (i // batch_size) % 50 == 0:
            pct = (i / len(images)) * 100
            print(f"  {i}/{len(images)} ({pct:.0f}%)")

    # Save scores
    print(f"\nScored {len(scores)} images")
    with open(SCORES_FILE, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Saved scores to {SCORES_FILE}")

    # Sort by score and take top N
    sorted_images = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Print score distribution
    all_scores = [s for _, s in sorted_images]
    print(f"\nScore distribution:")
    print(f"  Max:    {all_scores[0]:.4f}")
    print(f"  Top 10%: {all_scores[len(all_scores)//10]:.4f}")
    print(f"  Median: {all_scores[len(all_scores)//2]:.4f}")
    print(f"  Bottom 10%: {all_scores[-(len(all_scores)//10)]:.4f}")
    print(f"  Min:    {all_scores[-1]:.4f}")

    # Take top TARGET_COUNT
    selected = sorted_images[:TARGET_COUNT]
    threshold = selected[-1][1]
    print(f"\nSelecting top {TARGET_COUNT} images (score threshold: {threshold:.4f})")

    # Copy to clean dataset
    clean_images = OUTPUT_DIR / "images"
    clean_labels = OUTPUT_DIR / "labels"
    clean_images.mkdir(parents=True, exist_ok=True)
    clean_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_name, score in selected:
        src_img = DATASET_DIR / img_name
        src_label = LABELS_DIR / img_name.replace('.jpg', '.txt')

        if src_img.exists():
            shutil.copy2(src_img, clean_images / img_name)
            if src_label.exists():
                shutil.copy2(src_label, clean_labels / img_name.replace('.jpg', '.txt'))
            copied += 1

    print(f"\nCopied {copied} images + labels to {OUTPUT_DIR}/")

    # Show some examples
    print("\n=== TOP 10 (most billboard-like) ===")
    for name, score in sorted_images[:10]:
        print(f"  {score:.4f}  {name}")

    print("\n=== BOTTOM 10 (least billboard-like) ===")
    for name, score in sorted_images[-10:]:
        print(f"  {score:.4f}  {name}")

    print("\n=== AROUND THE CUTOFF ===")
    cutoff_idx = TARGET_COUNT
    for name, score in sorted_images[cutoff_idx-5:cutoff_idx+5]:
        marker = "KEEP" if score >= threshold else "DROP"
        print(f"  {score:.4f}  {name}  [{marker}]")


if __name__ == "__main__":
    main()
