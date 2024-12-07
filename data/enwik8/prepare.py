"""
Prepare the enwik8 dataset for character-level language modeling.
Split into:
- Train: first 90M characters
- Dev/Val: next 5M characters 
- Test: final 5M characters
"""
import os
import pickle
import requests
import numpy as np
from pathlib import Path

def download_and_extract():
    """Download and extract the enwik8 dataset"""
    try:
        data_url = 'http://mattmahoney.net/dc/enwik8.zip'
        r = requests.get(data_url)
        r.raise_for_status()
        
        zip_path = Path(__file__).parent / 'enwik8.zip'
        with open(zip_path, 'wb') as f:
            f.write(r.content)

        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Path(__file__).parent)
            
        # Clean up zip file
        zip_path.unlink()
        
    except Exception as e:
        print(f"Error downloading/extracting dataset: {e}")
        raise

def main():
    # download the enwik8 dataset
    input_file_path = Path(__file__).parent / 'enwik8'
    if not input_file_path.exists():
        download_and_extract()

    with open(input_file_path, 'r', encoding='latin-1') as f:
        data = f.read()
    
    # Verify data length is 100M
    assert len(data) == 100000000, f"Expected 100M characters, got {len(data):,}"
    print(f"length of dataset in characters: {len(data):,}")

    # Define split sizes
    TRAIN_SIZE = 90000000  # 90M
    VAL_SIZE = 5000000     # 5M
    TEST_SIZE = 5000000    # 5M
    
    assert TRAIN_SIZE + VAL_SIZE + TEST_SIZE == len(data), "Split sizes must sum to dataset size"

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s): return [stoi[c] for c in s]
    def decode(l): return ''.join([itos[i] for i in l])

    # create the train, validation, and test splits
    train_data = data[:TRAIN_SIZE]  # First 90M characters
    val_data = data[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]  # Next 5M characters
    test_data = data[TRAIN_SIZE+VAL_SIZE:]  # Final 5M characters

    # Verify split sizes
    assert len(train_data) == TRAIN_SIZE, f"Train split should be 90M but got {len(train_data)}"
    assert len(val_data) == VAL_SIZE, f"Val split should be 5M but got {len(val_data)}"
    assert len(test_data) == TEST_SIZE, f"Test split should be 5M but got {len(test_data)}"

    # encode all splits to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    test_ids = encode(test_data)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # export to bin files
    out_dir = Path(__file__).parent
    np.array(train_ids, dtype=np.uint16).tofile(out_dir / 'train.bin')
    np.array(val_ids, dtype=np.uint16).tofile(out_dir / 'val.bin')
    np.array(test_ids, dtype=np.uint16).tofile(out_dir / 'test.bin')

    # save the meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(out_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print("\nSplit sizes:")
    print(f"Train: {len(train_data):,} characters")
    print(f"Val:   {len(val_data):,} characters")
    print(f"Test:  {len(test_data):,} characters")

if __name__ == '__main__':
    main()