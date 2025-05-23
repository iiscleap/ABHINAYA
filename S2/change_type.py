import numpy as np
import os
from tqdm import tqdm

input_folder = "/data2/soumyad/SALMONN/audio_features_ft"
input_feats = os.listdir(input_folder)
input_feats = [x for x in input_feats if ".npy" in x]

for i, f in enumerate(tqdm(input_feats)):
    old_feat = np.load(os.path.join(input_folder, f))
    new_feat = old_feat.astype(np.float16)
    np.save(os.path.join(input_folder, f), new_feat)