import os
import shutil
import pandas as pd
from tqdm import tqdm
from random import sample
import random


def reorganize_images(source_dirs, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for source_dir in source_dirs:
        for person_id in tqdm(os.listdir(source_dir), desc=f"Processing {source_dir}"):
            person_dir = os.path.join(source_dir, person_id)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    source_image_path = os.path.join(person_dir, image_name)
                    dest_image_path = os.path.join(
                        dest_dir, person_id + "_" + image_name
                    )
                    shutil.copy(source_image_path, dest_image_path)


def create_pairs(images, max_pairs_per_person=200):
    positive_pairs = []
    negative_pairs = []
    labels = []
    person_to_images = {}

    for img in images:
        person_id = img.split("_")[0]
        if person_id not in person_to_images:
            person_to_images[person_id] = []
        person_to_images[person_id].append(img)

    for person_id, imgs in tqdm(
        person_to_images.items(), desc="Creating positive pairs"
    ):
        if len(imgs) > 1:
            for _ in range(min(max_pairs_per_person, len(imgs))):
                pair = random.sample(imgs, 2)
                positive_pairs.append(pair)
                labels.append(1)

    all_person_ids = list(person_to_images.keys())
    for _ in tqdm(range(len(positive_pairs)), desc="Creating negative pairs"):
        person1, person2 = random.sample(all_person_ids, 2)
        img1 = random.choice(person_to_images[person1])
        img2 = random.choice(person_to_images[person2])
        negative_pairs.append([img1, img2])
        labels.append(0)

    pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    combined = list(zip(pairs, labels))
    random.shuffle(combined)
    pairs, labels = zip(*combined)

    return list(pairs), list(labels)


source_dirs = ["data/face_similarity/train", "data/face_similarity/val"]
dest_dir = "data/rebuilded_face_similarity_data/images"

reorganize_images(source_dirs, dest_dir)

images = os.listdir(dest_dir)

max_images = 100000
if len(images) > max_images:
    images = images[:max_images]

pairs, labels = create_pairs(images)

split_index = int(len(pairs) * 0.8)
train_pairs, train_labels = pairs[:split_index], labels[:split_index]
val_pairs, val_labels = pairs[split_index:], labels[split_index:]

train_df = pd.DataFrame(train_pairs, columns=["image1", "image2"])
train_df["label"] = train_labels
train_df.to_csv("data/rebuilded_face_similarity_data/train_pairs.csv", index=False)

val_df = pd.DataFrame(val_pairs, columns=["image1", "image2"])
val_df["label"] = val_labels
val_df.to_csv("data/rebuilded_face_similarity_data/val_pairs.csv", index=False)
