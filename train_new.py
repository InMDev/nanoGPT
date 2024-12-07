import os
import time
import math
import pickle
import json  # For logging metrics
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import both models
import model as baseline_model_module
import model_new as spatial_model_module

# -----------------------------------------------------------------------------
# Configuration
# Common configurations for both models
out_dir = 'out'
dataset = 'enwik8'
batch_size = 12
block_size = 1024
eval_interval = 500  # Evaluate every 500 iterations
eval_iters = 100  # Use 100 iterations for evaluation
max_iters = 5000  # Adjust as needed
learning_rate = 6e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'  # Use float32 for simplicity
compile = False  # Set to True if using PyTorch 2.0 and want to compile the model

# Model-specific configurations
baseline_config = dict(n_layer=6, n_head=8, n_embd=256, dropout=0.1, bias=True)
spatial_config = dict(n_layer=6, n_head=8, n_embd=256, dropout=0.1, bias=True)
# -----------------------------------------------------------------------------

# Initialize distributed training if needed
ddp = False  # Set to True if using distributed data parallel
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else:
    device = device

# Set random seed for reproducibility
torch.manual_seed(42)

# Data loading
data_dir = os.path.join('data', dataset)

def get_batch(split):
    data_file = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))  # -1 to prevent overflow
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

# Load the vocabulary size from meta.pkl
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"Vocab size loaded from metadata: {vocab_size}")
else:
    raise FileNotFoundError(f"Metadata file not found at {meta_path}")

# Initialize both models
# Baseline model
baseline_model_args = dict(**baseline_config, vocab_size=vocab_size, block_size=block_size)
baseline_gpt_config = baseline_model_module.GPTConfig(**baseline_model_args)
baseline_model = baseline_model_module.GPT(baseline_gpt_config).to(device)

# Spatial model
spatial_model_args = dict(**spatial_config, vocab_size=vocab_size, block_size=block_size)
spatial_gpt_config = spatial_model_module.GPTConfig(**spatial_model_args)
spatial_model = spatial_model_module.GPT(spatial_gpt_config).to(device)

# Optionally compile the models (requires PyTorch 2.0)
if compile:
    baseline_model = torch.compile(baseline_model)
    spatial_model = torch.compile(spatial_model)

# Optimizers
baseline_optimizer = baseline_model.configure_optimizers(weight_decay=1e-2, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
spatial_optimizer = spatial_model.configure_optimizers(weight_decay=1e-2, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)

# Training settings
train_iters = max_iters
eval_interval = eval_interval

# Initialize logs for metrics
baseline_metrics = {'train_loss': [], 'val_loss': [], 'bpc': [], 'iter': []}
spatial_metrics = {'train_loss': [], 'val_loss': [], 'bpc': [], 'iter': []}

# Training loop
for iter_num in range(1, train_iters + 1):

    # Get training batch
    X_train, Y_train = get_batch('train')

    # Baseline model training step
    baseline_model.train()
    baseline_optimizer.zero_grad(set_to_none=True)
    logits, loss = baseline_model(X_train, Y_train)
    loss.backward()
    baseline_optimizer.step()
    baseline_train_loss = loss.item()

    # Spatial model training step
    spatial_model.train()
    spatial_optimizer.zero_grad(set_to_none=True)
    logits_spatial, loss_spatial = spatial_model(X_train, Y_train)
    loss_spatial.backward()
    spatial_optimizer.step()
    spatial_train_loss = loss_spatial.item()

    # Every eval_interval iterations, evaluate both models
    if iter_num % eval_interval == 0 or iter_num == 1:
        baseline_model.eval()
        spatial_model.eval()
        with torch.no_grad():
            # Evaluate baseline model
            baseline_val_losses = []
            for _ in range(eval_iters):
                X_val, Y_val = get_batch('val')
                logits, loss = baseline_model(X_val, Y_val)
                baseline_val_losses.append(loss.item())
            baseline_val_loss = np.mean(baseline_val_losses)
            baseline_bpc = baseline_val_loss / math.log(2)

            # Evaluate spatial model
            spatial_val_losses = []
            for _ in range(eval_iters):
                X_val, Y_val = get_batch('val')
                logits_spatial, loss_spatial = spatial_model(X_val, Y_val)
                spatial_val_losses.append(loss_spatial.item())
            spatial_val_loss = np.mean(spatial_val_losses)
            spatial_bpc = spatial_val_loss / math.log(2)

            # Log metrics
            baseline_metrics['train_loss'].append(baseline_train_loss)
            baseline_metrics['val_loss'].append(baseline_val_loss)
            baseline_metrics['bpc'].append(baseline_bpc)
            baseline_metrics['iter'].append(iter_num)

            spatial_metrics['train_loss'].append(spatial_train_loss)
            spatial_metrics['val_loss'].append(spatial_val_loss)
            spatial_metrics['bpc'].append(spatial_bpc)
            spatial_metrics['iter'].append(iter_num)

            # Print metrics
            print(f"Iteration {iter_num}:")
            print(f"Baseline Model - Train Loss: {baseline_train_loss:.4f}, Val Loss: {baseline_val_loss:.4f}, BPC: {baseline_bpc:.4f}")
            print(f"Spatial Model  - Train Loss: {spatial_train_loss:.4f}, Val Loss: {spatial_val_loss:.4f}, BPC: {spatial_bpc:.4f}")
            print("-" * 50)

# Final evaluation on test set
baseline_model.eval()
spatial_model.eval()
with torch.no_grad():
    # Evaluate baseline model on test set
    baseline_test_losses = []
    for _ in range(eval_iters):
        X_test, Y_test = get_batch('test')
        logits, loss = baseline_model(X_test, Y_test)
        baseline_test_losses.append(loss.item())
    baseline_test_loss = np.mean(baseline_test_losses)
    baseline_test_bpc = baseline_test_loss / math.log(2)

    # Evaluate spatial model on test set
    spatial_test_losses = []
    for _ in range(eval_iters):
        X_test, Y_test = get_batch('test')
        logits_spatial, loss_spatial = spatial_model(X_test, Y_test)
        spatial_test_losses.append(loss_spatial.item())
    spatial_test_loss = np.mean(spatial_test_losses)
    spatial_test_bpc = spatial_test_loss / math.log(2)

    # Log final test BPC
    print("Final Evaluation on Test Set:")
    print(f"Baseline Model - Test Loss: {baseline_test_loss:.4f}, BPC: {baseline_test_bpc:.4f}")
    print(f"Spatial Model  - Test Loss: {spatial_test_loss:.4f}, BPC: {spatial_test_bpc:.4f}")

# Save the models
torch.save(baseline_model.state_dict(), os.path.join(out_dir, 'baseline_model.pt'))
torch.save(spatial_model.state_dict(), os.path.join(out_dir, 'spatial_model.pt'))

# Save the metrics to JSON files for plotting
metrics_dir = os.path.join(out_dir, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)
with open(os.path.join(metrics_dir, 'baseline_metrics.json'), 'w') as f:
    json.dump(baseline_metrics, f)
with open(os.path.join(metrics_dir, 'spatial_metrics.json'), 'w') as f:
    json.dump(spatial_metrics, f)