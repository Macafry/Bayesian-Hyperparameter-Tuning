import polars as pl
import numpy as np
import os, shutil

def polars_train_test_split(df, test_size=0.2, random_state=None):
    """
    Splits a Polars DataFrame into training and testing sets.

    Parameters:
    df (pl.DataFrame): The input DataFrame to split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    train_df (pl.DataFrame): The training DataFrame.
    test_df (pl.DataFrame): The testing DataFrame.
    """
    # Shuffle the DataFrame
    df = df.sample(n=df.height, shuffle=True, seed=random_state)
    
    # Calculate the cutoff index for the test set
    cutoff = int(len(df) * (1 - test_size))
    train_df = df[:cutoff]
    test_df = df[cutoff:]

    return train_df, test_df

def file_train_test_split(file, out_dir, test_size=0.2, random_state=None):
    """
    Splits a CSV or Parquet file into training and testing sets and writes them to disk.
    """
    # Read file
    if file.endswith(".csv"):
        df = pl.read_csv(file)
    elif file.endswith(".parquet"):
        df = pl.read_parquet(file)
    else:
        raise ValueError("Unsupported file format: must be .csv or .parquet")

    # Shuffle
    df = df.sample(n=df.height, shuffle=True, seed=random_state)

    # Split
    cutoff = int(df.height * (1 - test_size))
    train_df, test_df = df[:cutoff], df[cutoff:]

    # Ensure output directory exists
    split_dir = os.path.join(out_dir, "train_test_split")
    os.makedirs(split_dir, exist_ok=True)

    # Save
    train_path = os.path.join(split_dir, "train.csv")
    test_path = os.path.join(split_dir, "test.csv")
    train_df.write_csv(train_path)
    test_df.write_csv(test_path)

    return train_path, test_path
    

def file_subsample(file, out_dir, sample_fraction=0.3, random_state=None):
    """
    Subsamples a Parquet or CSV file and saves the result as a CSV.

    Parameters:
        file (str): Path to the input CSV or Parquet file.
        out_dir (str): Directory to save the subsampled file.
        sample_fraction (float): Proportion of rows to keep (0 < f ≤ 1).
        random_state (int, optional): Seed for reproducibility.

    Returns:
        str: Path to the saved subsampled CSV file.
    """
    # Use lazy sampling for efficient reading
    if file.endswith(".parquet"):
        df = pl.read_parquet(file)
    elif file.endswith(".csv"):
        df = pl.read_csv(file)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    n = df.height
    subsample_df = df.sample(n=int(n*sample_fraction), with_replacement=False, seed=random_state)

    # Save to disk
    os.makedirs(out_dir, exist_ok=True)
    subsample_file = os.path.join(out_dir, "subsample.csv")
    subsample_df.write_csv(subsample_file)

    return subsample_file



def csv_no_headers(file, output=None):
    """
    Remove the header from a large CSV file efficiently.

    Args:
        file (str): Path to the input CSV file.
        output (str, optional): Output path. Defaults to file with _no_header suffix.

    Returns:
        str: Path to the saved CSV file without headers.
    """
    if output is None:
        output = file.replace(".csv", "_no_header.csv")
    
    with open(file, "r", encoding="utf-8") as infile, \
        open(output, "w", encoding="utf-8") as outfile:
        next(infile)
        shutil.copyfileobj(infile, outfile, length=1024*1024*4)
    
    return output
    
    
    