import pandas as pd
import numpy as np

train_pairs = pd.read_csv("data/rebuilded_face_simm/train_pairs.csv")
val_pairs = pd.read_csv("data/rebuilded_face_simm/val_pairs.csv")

train_pairs.head(5)

train_pairs["label"].value_counts()
val_pairs["label"].value_counts()