import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from src.utils.paths import paths
from . import preprocessers as pfunc
from src.utils.schema import Config

# Enable tqdm for pandas
tqdm.pandas()

def main():
    config = Config.load(paths.USER_CONFIG)
    print('\u2500' * 100)

    dataset_path = Path(config.dataset.raw_path)
    text_col_idx = config.dataset.text_column_index
    label_col_idx = config.dataset.label_column_index
    nrows: int | None = config.dataset.nrows_preprocess

    # Read data with specific columns by index
    df = pd.read_csv(dataset_path, usecols=[text_col_idx, label_col_idx], nrows=nrows, header=None)
    logger.debug(df.sample(3))
    logger.info(f"Loaded {len(df)} rows using col indices {text_col_idx} and {label_col_idx}")
    
    full_cols = pd.read_csv(dataset_path, nrows=0).columns
    
    print("CSV columns with indices:")
    for i, c in enumerate(full_cols):
        print(i, "→", c)
    print("usecols mapping:")
    print(text_col_idx, "→", full_cols[text_col_idx])
    print(label_col_idx, "→", full_cols[label_col_idx])

    # Rename them properly
    mapping = {
        text_col_idx: "cleaned_text",
        label_col_idx: "sentiment"
    }
    df = df.rename(columns=mapping)
    logger.info(mapping)
    # Clean
    orig = len(df)
    df = (df
        .dropna(subset=["cleaned_text"])
        .drop_duplicates(subset=["cleaned_text"])
        .reset_index(drop=True)
    )
    logger.info(f"Removed {orig - len(df)} duplicates/nulls")

    # Fast steps (no progress bar needed)
    fast_steps = [
        ("Removing HTML", pfunc.remove_html_tags),
        ("Cleaning text", pfunc.clean_text),
        ("Converting emojis", pfunc.convert_emojis_to_text),
        ("Filtering English", pfunc.filter_english_ascii),
        ("Replacing punctuation", pfunc.replace_punctuation_with_space),
        ("Removing whitespace", pfunc.remove_extra_whitespace),
        ("Lowercasing", pfunc.convert_to_lowercase),
    ]

    for name, step in fast_steps:
        df["cleaned_text"] = df["cleaned_text"].apply(step)
        logger.info(f"{name}")

    # Slow steps (with progress bar for each row)
    slow_steps = [
        ("Expanding contractions", pfunc.expand_contractions),
        ("Expanding slang", pfunc.expand_slang),
        ("Cleaning repeats", pfunc.clean_single_letters_and_repeats),
        ("Removing stopwords", pfunc.remove_stopwords),
    ]

    for name, step in slow_steps:
        logger.info(f"→ {name}...", "blue")
        df["cleaned_text"] = df["cleaned_text"].progress_apply(step)

    # Save
    output_path = Path(config.dataset.preprocessed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved to {output_path}")
    print('\u2500' * 100)

if __name__ == "__main__":
    main()