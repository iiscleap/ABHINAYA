import os
import torch
import torchaudio
import librosa
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam, AdamW
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import argparse
import time
from models.salmonn import SALMONN
from argparse import Namespace
from transformers import WhisperFeatureExtractor
import pandas as pd
from config import Config
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import torch.nn.functional as F

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

args_model = Namespace(cfg_path='configs/decode_config.yaml', device='cuda:0', options=None)
cfg = Config(args_model)
wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)


class SpeechDataset(Dataset):
    def __init__(
        self,
        details,
        mode="train"
    ):
        self.files = details["path"]
        self.files = [x for x in self.files if ".wav" in x]
        self.mode = mode
        if self.mode != "test":
            self.labels = details["label"]
            self.labels_dict = dict(zip(self.files, self.labels))
            self.final_labels = {}
            for k in self.labels_dict:
                if self.labels_dict[k] != "O" and self.labels_dict[k] != "X":
                    self.final_labels[k] = self.labels_dict[k]
            self.wav_files = list(self.final_labels.keys())
        else:
            self.wav_files = list(self.files)
        self.sr = 16000
        self.duration = 5000
        
        self.label_map = {"A":0, "C":1, "D":2, "F":3, "H":4, "N":5, "S":6, "U":7}

    def __len__(self):
        return len(self.wav_files)
    
    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)
        labels = torch.tensor([s["labels"] for s in samples])
        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
        wav_name = [s["wav_name"] for s in samples]


        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "labels": labels,
            "wav_name": wav_name
        }

    def __getitem__(self, index):
        wav_name = self.wav_files[index]
        if self.mode != "test":
            label = self.label_map[self.final_labels[wav_name]]
        else:
            label = -1

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
        spectrogram = wav_processor(final_sig, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        spectrogram = spectrogram.to(args_model.device)
        
        return {
            "spectrogram": spectrogram,
            "raw_wav": final_sig,
            "labels": label,
            "wav_name": wav_name.split(os.sep)[-1]
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


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
    def compute_length_from_mask(self, mask):
        """
        mask: (batch_size, T)
        Assuming that the sampling rate is 16kHz, the frame shift is 20ms
        """
        wav_lens = torch.sum(mask, dim=1) # (batch_size, )
        feat_lens = torch.div(wav_lens-1, 16000*0.02, rounding_mode="floor") + 1
        feat_lens = feat_lens.int().tolist()
        return feat_lens
        
    def forward(self, x, mask):
        raise NotImplementedError
    

class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    Link: https://arxiv.org/pdf/1803.10963.pdf
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs, mask):
        """
        xs: (batch_size, T, feat_dim)
        mask: (batch_size, T)

        => output: (batch_size, feat_dim*2)
        """
        # feat_lens = self.compute_length_from_mask(mask)
        pooled_list = []
        for x in xs:
            x = x.unsqueeze(0)
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            mu = torch.sum(x * w, dim=1)
            rh = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, rh), 1).squeeze(0)
            pooled_list.append(x)
        return torch.stack(pooled_list)



class EmotionClassifier(nn.Module):
    def __init__(self,
                 salmonn_model,
                 hidden_dim,
                 output_dim):
        
        super().__init__()
        self.salmonn_model = salmonn_model
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc = nn.Linear(4096*2, hidden_dim)
        self.pool = AttentiveStatisticsPooling(4096)
        self.ln = nn.LayerNorm(4096)


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def get_features(self, samples):

        feats_final = self.salmonn_model(samples)["hidden_states"][-1, :, :, :]

        feats_final = self.ln(feats_final)

        return feats_final

        
    def forward(self, samples, audio):
        samples["text"] = ["Describe emotion of the speaker in one word"]*audio.shape[0]
        samples["task"] = ["emotion_recognition"]*audio.shape[0]

        feats_final = self.salmonn_model(samples)["hidden_states"][-1, :, :, :]
        feats_final = self.ln(feats_final)
        feat = self.pool(feats_final, ~samples["padding_mask"])

        feat = self.dropout(self.relu(self.fc(feat)))
        
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

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, pred, target):
        output = pred / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)

def compute_accuracy(output, labels):
    #Function for calculating accuracy
    pred = torch.argmax(output, dim = 1)
    correct_pred = (pred == labels).float()
    tot_correct = correct_pred.sum()

    return tot_correct

def compute_loss(output, labels):
    #Function for calculating loss

    ce_loss = nn.CrossEntropyLoss(reduction='none')(output, labels.squeeze(-1).long())
    pt = torch.exp(-ce_loss)
    loss = ((1-pt)**0 * ce_loss).mean()
    return loss

def create_dataset(mode, bs=8):
    if mode == 'train':
        f = open("/data1/soumyad/IS2025_challenge/train_dict.json")
        details = json.load(f)
        f.close()
        mode = "train"
    elif mode == 'val':
        f = open("/data1/soumyad/IS2025_challenge/balanced_valid_final.json")
        details = json.load(f)
        f.close()
        mode = "train"
    else:
        f = open("/data1/soumyad/IS2025_challenge/test_dict.json")
        details = json.load(f)
        f.close()
        mode = "test"
    dataset = SpeechDataset(details, mode)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=dataset.collater)
    return loader
    
def train():

    train_loader = create_dataset("train", 8)
    val_loader = create_dataset("val", 8)
    num_classes = 8
    class_weights = torch.tensor([1.25, 3.125, 6.25, 6.25, 0.5, 0.28, 1.38, 3.125]).to(device)
    class_numbers = np.array([6728, 2495, 1432, 1120, 16712, 29239, 6303, 2948])

    alpha = 1.0
    gamma = 2.0
    criterion = WeightedFocalLoss(alpha=alpha, gamma=gamma, weight=class_weights)
    criterion_nowt = WeightedFocalLoss(alpha=alpha, gamma=gamma)
    
    salmonn_model = SALMONN.from_config(cfg.config.model)
    model = EmotionClassifier(salmonn_model, 1024, num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    base_lr = 1e-5
    optimizer = Adam(
            params,
            lr=base_lr
        )
    final_val_loss = 0
    accumulation_steps = 4
    scaler = torch.cuda.amp.GradScaler() 
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
            aud, spectrogram, labels, padding_mask = data["raw_wav"].to(device), data["spectrogram"].to(device), data["labels"].to(device), data["padding_mask"].to(device)
            samples = {"spectrogram": spectrogram,"raw_wav": aud,"padding_mask": padding_mask}
            with torch.cuda.amp.autocast(enabled=True):
                final_out = model(samples, aud)
                loss = criterion(final_out, labels) / accumulation_steps
            scaler.scale(loss).backward() 
            tot_loss += loss.detach().item()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            pred = torch.argmax(final_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_size += data["raw_wav"].shape[0]
                aud, spectrogram, labels, padding_mask = data["raw_wav"].to(device), data["spectrogram"].to(device), data["labels"].to(device), data["padding_mask"].to(device)
                samples = {"spectrogram": spectrogram,"raw_wav": aud,"padding_mask": padding_mask}
                with torch.cuda.amp.autocast(enabled=True):
                    val_out = model(samples, aud)
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
            state_dict = model.state_dict()
            param_grad_dic = {
            k: v.requires_grad for (k, v) in model.named_parameters()
            }
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    # delete parameters that do not require gradient
                    del state_dict[k]
            torch.save(state_dict, "salmonn_podcast_7b.pth")
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
    test_loader = create_dataset("test", 1)
    num_classes = 8
    salmonn_model = SALMONN.from_config(cfg.config.model)
    model = EmotionClassifier(salmonn_model, 1024, num_classes)
    model.load_state_dict(torch.load("salmonn_podcast_statpool.pth"), strict=False)
    model.to(device)
    model.eval()
    f = open("/data1/soumyad/IS2025_challenge/test_transcripts.json")
    details = json.load(f)
    f.close()
    files = details["path"]
    label_map = {0:"A", 1:"C", 2:"D", 3:"F", 4:"H", 5:"N", 6:"S", 7:"U"}
    predicted_dict = {"FileName":[], "EmoClass":[]}
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            aud, spectrogram, target, padding_mask = data["raw_wav"].to(device), data["spectrogram"].to(device), data["labels"].to(device), data["padding_mask"].to(device)
            samples = {"spectrogram": spectrogram,"raw_wav": aud,"padding_mask": padding_mask}
            wav_name = data["wav_name"][0]
            with torch.cuda.amp.autocast(enabled=True):
                test_out = model(samples, aud)
            pred = torch.argmax(test_out, dim = 1)
            pred = pred.detach().cpu().numpy()[0]
            predicted_dict["FileName"].append(wav_name)
            predicted_dict["EmoClass"].append(label_map[pred])
    df = pd.DataFrame(predicted_dict)
    df.to_csv("salmonn_statpool_test.csv", index=False)



if __name__ == "__main__":
    train()
    test()
