# train_cnn.py
import os
import argparse
import numpy as np
import pandas as pd
import uproot
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

NEG_DIR = "/projects/cos-lab-aotte6/cos-4a17118/Trinity/TrinityDemonstratorDataAnalysis/event_cleaning/Output/"
POS_DIR = "/projects/cos-lab-aotte6/cos-4a17118/Trinity/TrinityDemonstrator/simdata/sofiadata/20260924_taudavid"

# (Paste your CNN class, CherenkovDataset class, train_model, and evaluate_and_save functions here)
# Note: In train_model() and evaluate_and_save(), remove `tqdm(...)` or pass `disable=True` 
# so your Condor .err log file doesn't fill up with thousands of progress bar lines!
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        """
        Simplified Convolutional Neural Network for 18x18 IACT image data[cite: 2].

        Parameters:
            * in_channels: Number of channels in input image
            * num_classes: Number of classes to predict (binary classifier = 2)
        """
        super(CNN, self).__init__()

        # 1st Convolutional layer: 32 out channels, 3x3 kernel, 0 padding
        # 18x18 input -> 16x16 output after Conv2d
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=0)
        
        # Max pooling layer 1
        # 16x16 -> 8x8 output after MaxPool2d
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd Convolutional layer: 64 out channels, 3x3 kernel, 1 padding
        # 8x8 -> 8x8 output after Conv2d
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max pooling layer 2
        # 8x8 -> 4x4 output after MaxPool2d
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layers (64 channels * 4 * 4 = 1024 nodes)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.
        """
        x = F.relu(self.conv1(x))       # (Batch, 32, 16, 16)
        x = self.pool1(x)               # (Batch, 32, 8, 8)
        
        x = F.relu(self.conv2(x))       # (Batch, 64, 8, 8)
        x = self.pool2(x)               # (Batch, 64, 4, 4)
        
        x = self.flatten(x)             # Flatten to 1024 nodes
        x = F.relu(self.fc1(x))         # First FC layer (512 nodes)
        
        # Note: Softmax is omitted here because standard PyTorch training 
        # uses nn.CrossEntropyLoss, which applies Softmax internally.
        x = self.fc2(x)                 # Output layer
        return x

class CherenkovDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        pos_dir, 
        neg_dir, 
        tree_name="Events/T0",       # Matches the TDirectory/TTree structure in your sample .root file
        branch_name="vPEInPixel",    # Or "vQDCValue", "vSumTimeInPixel", etc.
        pos_event_range=(1, None),   # 1-indexed (start_event, end_event) tuple, e.g., (1, 5000)
        neg_event_range=(1, None),   # 1-indexed (start_event, end_event) tuple
        img_shape=(1, 18, 18)        # Reshapes flat pixel arrays into (Channels, Height, Width)
    ):
        """
        Recursively finds .root files using os.walk, extracts a configurable 1-to-n 
        range of events, and labels them: Positive (1) and Negative (0).
        """
        self.tree_name = tree_name
        self.branch_name = branch_name
        self.img_shape = img_shape

        # Load positive and negative datasets across arbitrary folder depths
        pos_data = self._load_from_directory(pos_dir, pos_event_range)
        neg_data = self._load_from_directory(neg_dir, neg_event_range)

        # Combine datasets and create binary labels (1 = Cherenkov flash, 0 = Noise)
        self.data = np.concatenate([pos_data, neg_data], axis=0)
        self.labels = np.concatenate([
            np.ones(len(pos_data), dtype=np.int64), 
            np.zeros(len(neg_data), dtype=np.int64)
        ], axis=0)

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _get_root_files(self, base_dir):
        """
        Uses os.walk to recursively gather all .root files regardless of folder nesting.
        Sorting guarantees deterministic event ordering for 1-to-n cross-validation splits.
        """
        if not base_dir or not os.path.exists(base_dir):
            print(f"Warning: Directory '{base_dir}' not found or is None. Returning empty dataset.")
            return []

        root_files = []
        for root, dirs, files in os.walk(base_dir):
            dirs.sort()  # Enforce deterministic directory traversal order
            for file_name in sorted(files):
                if file_name.endswith('.root'):
                    root_files.append(os.path.join(root, file_name))
                    
        return root_files

    def _load_from_directory(self, base_dir, event_range):
        root_files = self._get_root_files(base_dir)
        if not root_files:
            return np.empty((0, *self.img_shape), dtype=np.float32)

        # Convert 1-based inclusive range (1, n) to 0-based Python slice indices [start_idx, end_idx)
        start_idx = max(0, (event_range[0] - 1)) if event_range[0] is not None else 0
        end_idx = event_range[1] if event_range[1] is not None else float('inf')

        data_list = []
        current_event_idx = 0

        for file_path in root_files:
            if current_event_idx >= end_idx:
                break

            try:
                with uproot.open(file_path) as file:
                    if self.tree_name not in file:
                        continue
                    tree = file[self.tree_name]
                    num_entries = tree.num_entries

                    # If the entire file comes before our desired start_event, skip reading its arrays
                    if current_event_idx + num_entries <= start_idx:
                        current_event_idx += num_entries
                        continue

                    # Calculate exact entry slice inside this specific .root file
                    entry_start = max(0, start_idx - current_event_idx)
                    entry_stop = min(num_entries, end_idx - current_event_idx)

                    # Extract only the needed slice into a NumPy array
                    array = tree[self.branch_name].array(
                        entry_start=entry_start,
                        entry_stop=entry_stop,
                        library="np"
                    )

                    # Reshape flat detector pixel arrays (e.g., 256 or 324 pixels) into (N, C, 18, 18)
                    array = self._format_image_shape(array)
                    data_list.append(array)

                    current_event_idx += num_entries

            except Exception as e:
                print(f"Skipping {file_path} due to read error: {e}")

        if not data_list:
            return np.empty((0, *self.img_shape), dtype=np.float32)

        return np.concatenate(data_list, axis=0)

    def _format_image_shape(self, array):
        """
        Unpacks uproot object arrays (arrays of per-event 1D arrays) into a 
        contiguous 2D float32 matrix before padding and reshaping to (N, C, H, W).
        """
        num_samples = len(array)
        if num_samples == 0:
            return np.empty((0, *self.img_shape), dtype=np.float32)

        target_pixels = np.prod(self.img_shape)

        # If uproot returned an array of objects (1D array per event), stack or pad them
        if array.dtype == object:
            # Check if all events in this file have the exact same number of pixels
            lengths = [len( np.ravel(x) ) for x in array]
            if len(set(lengths)) == 1 and lengths[0] > 0:
                flat_array = np.vstack([np.ravel(x) for x in array]).astype(np.float32)
            else:
                # Jagged events (variable length per event): copy into a pre-allocated float32 matrix
                flat_array = np.zeros((num_samples, target_pixels), dtype=np.float32)
                for i, ev in enumerate(array):
                    ev_flat = np.asarray(ev, dtype=np.float32).ravel()
                    n_copy = min(len(ev_flat), target_pixels)
                    flat_array[i, :n_copy] = ev_flat[:n_copy]
                return flat_array.reshape(num_samples, *self.img_shape)
        else:
            flat_array = np.asarray(array, dtype=np.float32).reshape(num_samples, -1)

        # Pad or truncate uniform arrays to match target_pixels (e.g., 256 -> 324 for 18x18)
        current_pixels = flat_array.shape[1]
        if current_pixels < target_pixels:
            pad_width = target_pixels - current_pixels
            flat_array = np.pad(flat_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif current_pixels > target_pixels:
            flat_array = flat_array[:, :target_pixels]

        return flat_array.reshape(num_samples, *self.img_shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(model, dataloader, epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # tqdm for progress bar tracking
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=None)
        for batch_data, batch_labels in progress_bar:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"-> Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}\n")

def evaluate_and_save(model, dataloader, device='cpu', output_file="classification_results.csv"):
    model = model.to(device)
    model.eval()
    
    all_truths = []
    all_preds = []
    all_confs = []
    
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Evaluating", disable=None):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            outputs = model(batch_data)
            
            # Apply softmax to get probability distributions (confidences)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)
            
            all_truths.extend(batch_labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_confs.extend(confidences.cpu().numpy())

    # Build performance dataframe
    results_df = pd.DataFrame({
        'Ground_Truth': all_truths,
        'Assigned_Class': all_preds,
        'Confidence': all_confs
    })
    
    # Assess correctness
    results_df['Correctness'] = results_df['Ground_Truth'] == results_df['Assigned_Class']
    
    # Calculate overarching metrics
    accuracy = results_df['Correctness'].mean() * 100
    print(f"\nEvaluation Complete - Accuracy: {accuracy:.2f}%")
    
    # Exporting
    results_df.to_csv(output_file, index=False)
    print(f"Detailed performance metrics saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test Cherenkov CNN Classifier")
    parser.add_argument("--pos_dir", type=str, default="/projects/cos-lab-aotte6/cos-4a17118/Trinity/TrinityDemonstrator/simdata/sofiadata/20260924_taudavid")
    parser.add_argument("--neg_dir", type=str, default="/projects/cos-lab-aotte6/cos-4a17118/Trinity/TrinityDemonstratorDataAnalysis/event_cleaning/Output/")
    parser.add_argument("--train_start", type=int, default=1)
    parser.add_argument("--train_end", type=int, default=4000)
    parser.add_argument("--test_start", type=int, default=4001)
    parser.add_argument("--test_end", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out_csv", type=str, default="cnn_test_metrics.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_range = (args.train_start, args.train_end)
    test_range = (args.test_start, args.test_end)

    print(f"Loading training data: events {train_range}...")
    train_dataset = CherenkovDataset(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        pos_event_range=train_range,
        neg_event_range=train_range
    )

    print(f"Loading testing data: events {test_range}...")
    test_dataset = CherenkovDataset(
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir,
        pos_event_range=test_range,
        neg_event_range=test_range
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CNN(in_channels=1, num_classes=2).to(device)
    train_model(model, train_loader, epochs=args.epochs, lr=args.lr, device=device)
    
    # Save trained model weights alongside the CSV metrics
    torch.save(model.state_dict(), "cnn_model_weights.pth")
    evaluate_and_save(model, test_loader, device=device, output_file=args.out_csv)