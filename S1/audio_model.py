import os
import torch
import torchaudio
import librosa
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, SGD
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import argparse
import time
from transformers import AutoProcessor, WavLMModel, AutoConfig, WavLMConfig
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf

#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Fine tune")
parser.add_argument(
    "--seed",
    metavar="seed",
    type=int,
)


args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class SpeechDataset(Dataset):
    def __init__(
        self,
        details
    ):
        self.files = details["path"]
        self.labels = details["label"]
        self.files = [x for x in self.files if ".wav" in x]
        self.labels_dict = dict(zip(self.files, self.labels))
        self.final_labels = {}
        for k in self.labels_dict:
            if self.labels_dict[k] != "O" and self.labels_dict[k] != "X":
                self.final_labels[k] = self.labels_dict[k]
        self.sr = 16000
        self.duration = 10000
        self.wav_files = list(self.final_labels.keys())
        self.label_map = {"A":0, "C":1, "D":2, "F":3, "H":4, "N":5, "S":6, "U":7}

    def __len__(self):
        return len(self.wav_files)
    
    def collater(self, samples):
        labels = torch.tensor([s["labels"] for s in samples])
        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)


        return {
            "raw_wav": raw_wav.float(),
            "padding_mask": paddding_mask,
            "labels": labels,
        }

    def __getitem__(self, index):
        wav_name = self.wav_files[index]
        label = self.label_map[self.final_labels[wav_name]]

        audio_file = wav_name   
        (sig, sr) = sf.read(audio_file)

        aud = (sig, sr)
        reaud = (sig, self.sr)
        resig = sig
        sig_len = resig.shape[0]
        max_len = self.sr//1000 * self.duration
        if len(resig.shape) == 2:
            resig = np.mean(resig, axis = 1)

        if (sig_len > max_len):
            # Truncate the signal to the given length
            start = np.random.randint(0, sig_len-max_len)

            final_sig = resig[start:start+max_len]

        else:
            final_sig = resig
        
        return {
            "raw_wav": final_sig,
            "labels": label,
        }

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class EmotionClassifier(nn.Module):
    def __init__(self,
                 wavlm_model,
                 hidden_dim,
                 output_dim):
        
        super().__init__()
        self.wavlm_model = wavlm_model
        
        # self.weights = nn.Parameter(torch.rand(25, 1))
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(1024, hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.ln = nn.LayerNorm(1024)

    def get_features(self, aud):
        outputs = self.wavlm_model(aud, output_hidden_states=True, return_dict=True)
        feat = torch.stack(list(outputs['hidden_states']), dim=0)#.squeeze(1)
        feat = feat.permute(1, 3, 2, 0)
        weights_normalized = nn.functional.softmax(self.weights, dim=0)
        feats_final = torch.matmul(feat, weights_normalized.squeeze())
        feats_final = feats_final.permute(0, 2, 1)
        
        return feats_final
        
    def forward(self, aud, padding_mask):
        # print(aud.shape)
        outputs = self.wavlm_model(aud, output_hidden_states=True, return_dict=True, attention_mask=~padding_mask)
        feats_final = outputs['last_hidden_state']
        feat = self.fc(feat)
        feat = self.ln(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)
        output = self.out(feat)
        return output

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, weight=None):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight  # Optional: can be used for class weighting

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        softmax = torch.softmax(inputs, dim=1)
        
        # Get the probability of the true class for each sample
        p_t = softmax.gather(1, targets.unsqueeze(1))
        
        # Compute the focal loss part
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)

        # If weight is provided, apply it per class
        if self.weight is not None:
            weight = self.weight[targets]
            loss = loss * weight
        
        return loss.mean()

def compute_accuracy(output, labels):
    #Function for calculating accuracy
    pred = torch.argmax(output, dim = 1)
    correct_pred = (pred == labels).float()
    tot_correct = correct_pred.sum()

    return tot_correct

def create_dataset(mode, bs=8):
    if mode == 'train':
        f = open("/data1/soumyad/IS2025_challenge/train_dict.json")
        details = json.load(f)
        f.close()
    elif mode == 'val':
        f = open("/data1/soumyad/IS2025_challenge/balanced_valid_final.json")
        details = json.load(f)
        f.close()
    else:
        f = open("/data1/soumyad/IS2025_challenge/test_dict.json")
        details = json.load(f)
        f.close()
    dataset = SpeechDataset(details)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=dataset.collater)
    return loader
    
def train():

    train_loader = create_dataset("train", 16)
    val_loader = create_dataset("val", 16)
    num_classes = 8
    class_weights = torch.tensor([1.25, 3.125, 6.25, 6.25, 0.5, 0.28, 1.38, 3.125]).to(device)
    alpha = 1.0
    gamma = 2.0
    criterion = WeightedFocalLoss(alpha=alpha, gamma=gamma, weight=class_weights)
    

    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    
    model = EmotionClassifier(wavlm_model, 1024, num_classes)
    model.to(device)
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = 1e-5
    optimizer = Adam(
            params,
            lr=base_lr,
        )
    final_val_loss = 0

    for e in range(20):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        train_size = 0
        val_size = 0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(train_loader):
            train_size += data["raw_wav"].shape[0]
            # Get the input features and target labels, and put them on the GPU
            aud, labels, padding_mask = data["raw_wav"].to(device), data["labels"].to(device), data["padding_mask"].to(device)
            final_out = model(aud, padding_mask)
            # loss = compute_loss(final_out, labels)
            loss = criterion(final_out, labels)
            optimizer.zero_grad()
            loss.backward()
            tot_loss += loss.detach().item()
            optimizer.step()
            pred = torch.argmax(final_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_size += data["raw_wav"].shape[0]
                aud, labels, padding_mask = data["raw_wav"].to(device), data["labels"].to(device), data["padding_mask"].to(device)
                val_out = model(aud, padding_mask)
                loss = criterion(val_out, labels)
                val_loss += loss.item()
                pred = torch.argmax(val_out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        val_f1 = f1_score(gt_val, pred_val, average='macro')
        if val_f1 > final_val_loss:
            torch.save(model, "wavlm_model.pth")
            final_val_loss = val_f1
        train_loss = tot_loss/len(train_loader)
        train_f1 = f1_score(gt_tr, pred_tr, average='macro')
        val_loss_log = val_loss/len(val_loader)
        val_f1 = f1_score(gt_val, pred_val, average='macro')
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")
        
def test():

    val_loader = create_dataset("val", 1)
    num_classes = 8
    

    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    
    model = EmotionClassifier(wavlm_model, 1024, num_classes)
    model = torch.load("wavlm_large_weighted.pth")
    model.to(device)
    model.eval()
    pred_test, gt_test = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            aud, labels = data[0], data[1]
            aud = torch.stack(list(aud), dim=0).to(device)
            labels = torch.stack(list(labels), dim=0).to(device)
            test_out = model(aud)
            pred = torch.argmax(test_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_test.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_test.extend(labels)
    test_f1 = f1_score(gt_test, pred_test, average='macro')
    logger.info(f"Test Accuracy {test_f1}")

def inference():
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    num_classes = 8
    model = EmotionClassifier(wavlm_model, 1024, num_classes)
    model = torch.load("wavlm_large_weighted.pth")
    model.to(device)
    model.eval()
    f = open("/data1/soumyad/IS2025_challenge/test_transcripts.json")
    details = json.load(f)
    f.close()
    files = details["path"]
    label_map = {0:"A", 1:"C", 2:"D", 3:"F", 4:"H", 5:"N", 6:"S", 7:"U"}
    predicted_dict = {"FileName":[], "EmoClass":[]}
    with torch.no_grad():
        for i, f in enumerate(tqdm(files)):
            wav_name = f.split(os.sep)[-1]
            aud, _ = torchaudio.load(f)
            final_out = model(aud.to(device))
            pred = torch.argmax(final_out, dim = 1)
            pred = pred.detach().cpu().numpy()[0]
            predicted_dict["FileName"].append(wav_name)
            predicted_dict["EmoClass"].append(label_map[pred])
    df = pd.DataFrame(predicted_dict)
    df.to_csv("categorical_audio_weighted.csv", index=False)


if __name__ == "__main__":
    train()
    test()
    inference()
