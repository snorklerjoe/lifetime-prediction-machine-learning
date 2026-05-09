"""
Created by Gemini
This script generates synthetic time-series data using the TimeGAN model from the ydata-synthetic library.
Usage:
    python generate_timegan.py <input1.csv> [input2.csv ...] <output_prefix> <num_files>
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warn = print

from data_synthetic.synthesizers.timeseries import TimeGAN
from data_synthetic.synthesizers import ModelParameters, TrainParameters

def extract_sequences(data, seq_len):
    """
    Converts 2D data into 3D sequences of shape (samples, seq_len, features)
    using a sliding window approach.
    """
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i : i + seq_len])
    return np.array(sequences)

def main():
    # 1. Parse Arguments
    if len(sys.argv) < 4:
        print("Usage: python generate_timegan.py <input1.csv> [input2.csv ...] <output_prefix> <num_files>")
        sys.exit(1)

    # All arguments except the script name, the output prefix, and the count
    input_paths = sys.argv[1:-2]
    output_prefix = sys.argv[-2]
    
    try:
        num_files = int(sys.argv[-1])
    except ValueError:
        print("Error: The final argument (number of synthetic files) must be an integer.")
        sys.exit(1)

    print(f"Input files: {input_paths}")
    print(f"Output prefix: {output_prefix}")
    print(f"Number of files to generate: {num_files}")

    # 2. Load and Preprocess Data
    # Hyperparameter: Sequence length for the time-series windows
    seq_len = 24 
    
    all_data = []
    columns = None

    for path in input_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue
            
        df = pd.read_csv(path)
        # Drop non-numeric columns if necessary, or assume they are pre-processed
        df = df.select_dtypes(include=[np.number]) 
        
        if columns is None:
            columns = df.columns.tolist()
            
        all_data.append(df.values)

    if not all_data:
        print("Error: No valid input data found.")
        sys.exit(1)

    # Concatenate all input data
    combined_data = np.vstack(all_data)
    
    # Scale data to [0, 1] for better GAN training
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # Convert 2D scaled data into 3D sequences
    training_data = extract_sequences(scaled_data, seq_len)
    n_seq = training_data.shape[2] # Number of features
    
    print(f"Training data shape (samples, seq_len, features): {training_data.shape}")

    # 3. Setup and Train TimeGAN
    # Note: These are baseline parameters. You may need to tune these for your specific dataset.
    gan_args = ModelParameters(
        batch_size=128,
        lr=5e-4,
        noise_dim=32,
        layers_dim=128,
        latent_dim=24,
        gamma=1
    )
    
    train_args = TrainParameters(
        epochs=500, # Increase epochs for better results (e.g., 5000 - 10000) depending on data
        sequence_length=seq_len,
        number_sequences=n_seq
    )

    print("Initializing and training TimeGAN...")
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
    
    # Train the model (expects a list of 2D arrays or a 3D array)
    # ydata-synthetic expects a list of sequences
    synth.train(training_data.tolist(), train_args)
    print("Training complete.")

    # 4. Generate and Save Synthetic Data
    samples_to_generate = training_data.shape[0]

    for i in range(1, num_files + 1):
        print(f"Generating synthetic dataset {i}...")
        synth_data = synth.sample(samples_to_generate)
        
        # synth_data is 3D. We need to flatten it back to 2D for CSV storage.
        # We also inverse-transform it back to the original numerical scale.
        samples, seq, feats = synth_data.shape
        synth_2d = synth_data.reshape(-1, feats)
        synth_2d_unscaled = scaler.inverse_transform(synth_2d)
        
        # Create a DataFrame
        out_df = pd.DataFrame(synth_2d_unscaled, columns=columns)
        
        # Insert a Sequence_ID column to help identify distinct time-series sequences
        seq_ids = np.repeat(np.arange(samples), seq)
        out_df.insert(0, 'Sequence_ID', seq_ids)
        
        # Determine the final output filename
        out_name = f"{output_prefix}.{i}.csv"
        out_df.to_csv(out_name, index=False)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
