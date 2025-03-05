import os
import math
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
import pandas as pd
import gcsfs
import torchaudio
import librosa
import transformers
from transformers import (
    AutoModelForSpeechSeq2Seq,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import random
from transformers.modeling_outputs import BaseModelOutput
import wandb  # Using Weights & Biases for tracking
import soundfile as sf  # Using soundfile directly

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AudioDataset(Dataset):
    def __init__(self, gcs_bucket, csv_path, tokenizer, feature_extractor, split='train', max_audio_length=480000, original_sample_rate=48000, target_sample_rate=16000):
        self.gcs_bucket = gcs_bucket
        self.split = split
        self.max_audio_length = max_audio_length
        self.original_sample_rate = original_sample_rate
        self.target_sample_rate = target_sample_rate
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        
        # Load metadata from CSV
        self.labels_df = pd.read_csv(csv_path)
        self.file_id_to_caption = dict(zip(self.labels_df['ytid'].astype(str),
                                           self.labels_df['caption']))
        
        base_path = 'gs://musiccaps/excerpts/'
        self.audio_files = [f for f in self.gcs_bucket.ls(base_path) if f.endswith('.wav')]
        
        # Ensure only files in the bucket that exist in the CSV
        self.audio_files = [f for f in self.audio_files if os.path.splitext(os.path.basename(f))[0] in self.file_id_to_caption]
        
        # Split data into 80% training and 20% validation
        total_files = len(self.audio_files)
        val_size = int(0.2 * total_files)
        train_size = total_files - val_size
        self.train_files, self.val_files = random_split(self.audio_files, [train_size, val_size])
        
        # Select split
        self.audio_files = self.train_files if split == 'train' else self.val_files

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        wav_file = self.audio_files[idx]
        file_id = os.path.splitext(os.path.basename(wav_file))[0]
        
        with self.gcs_bucket.open(wav_file, 'rb') as f:
            file_bytes = f.read()
        buffer = BytesIO(file_bytes)
        data, orig_sr = sf.read(buffer, dtype='float32')
        
        if data is None or data.size == 0:
            print(f"Skipping empty audio file: {wav_file}")
            return self.__getitem__((idx + 1) % len(self.audio_files))
        
        if data.ndim == 2:
            data = data.mean(axis=1)  # Convert stereo to mono
        audio_tensor = torch.from_numpy(data)
        
        # Ensure audio_tensor is not empty
        if audio_tensor.numel() == 0:
            print(f"Skipping empty tensor for file: {wav_file}")
            return self.__getitem__((idx + 1) % len(self.audio_files))
        
        # Resample to 16kHz
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.target_sample_rate)
        audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        
        if audio_tensor.size(0) > self.max_audio_length:
            start_idx = torch.randint(0, audio_tensor.size(0) - self.max_audio_length + 1, (1,)).item()
            audio_tensor = audio_tensor[start_idx:start_idx + self.max_audio_length]
        else:
            audio_tensor = F.pad(audio_tensor, (0, self.max_audio_length - audio_tensor.size(0)))
        
        caption = self.file_id_to_caption.get(file_id)
        if caption is None:
            print(f"Skipping file with missing caption: {wav_file}")
            return self.__getitem__((idx + 1) % len(self.audio_files))
        
        # Tokenize caption properly
        labels = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0)
        
        # Extract features
        audio_features = self.feature_extractor(audio_tensor.numpy(), sampling_rate=self.target_sample_rate, return_tensors="pt").input_features.squeeze(0)
        
        return {
            "input_features": audio_features,
            "labels": labels
        }

# Custom Data Collator for Whisper
class WhisperDataCollator:
    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])
        labels = [f["labels"] for f in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_features": input_features, "labels": labels}

def main():
    set_seed(42)
    
    # Initialize Weights & Biases
    wandb.init(project="whisper-audio-captioning")
    
    # Set the multiprocessing start method.
    mp.set_start_method('spawn', force=True)
    
    # Set the torchaudio backend to "soundfile" so that it can read from file-like objects.
    torchaudio.set_audio_backend("soundfile")
    
    # Load model
    checkpoint = "MU-NLPC/whisper-large-v2-audio-captioning"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint)
    tokenizer = WhisperTokenizer.from_pretrained(checkpoint, language="en", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
    
    # Load dataset
    gcs = gcsfs.GCSFileSystem()
    data_path = "musiccaps-public.csv"
    dataset = AudioDataset(gcs, data_path, tokenizer, feature_extractor, split='train')
    val_dataset = AudioDataset(gcs, data_path, tokenizer, feature_extractor, split='val')
    
    # Define custom data collator
    data_collator = WhisperDataCollator()
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results/another_trial",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1000,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=20,
        logging_dir="./logs",
        logging_steps=100,
        save_total_limit=6,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True,
        report_to=["wandb"],
        dataloader_num_workers=8
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train model
    trainer.train()
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
