import sys
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import preprocessers as pfunc
from tqdm import tqdm
from loguru import logger
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from src.utils.paths import paths

# Enable tqdm for pandas
tqdm.pandas()

def print_info(msg, color="blue"):
    colors = {"blue": "\033[94m", "green": "\033[92m", "reset": "\033[0m"}
    print(f"{colors.get(color, '')}{msg}{colors['reset']}")

def main():
    config = OmegaConf.load(paths.USER_CONFIG)

    dataset_path = Path(config.dataset.raw_path)
    text_col_idx = config.dataset.text_column_index
    label_col_idx = config.dataset.label_column_index
    nrows = None if str(config.dataset.nrows_preprocess).lower() == "none" else int(config.dataset.nrows_preprocess)

    # Read data with specific columns by index
    df = pd.read_csv(dataset_path, usecols=[text_col_idx, label_col_idx], nrows=nrows, header=None)
    logger.debug(df.sample(3))
    print_info(f"✓ Loaded {len(df)} rows using col indices {text_col_idx} and {label_col_idx}", "green")
    
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
    print_info(f"✓ Removed {orig - len(df)} duplicates/nulls", "blue")

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
        print_info(f"✓ {name}", "blue")

    # Slow steps (with progress bar for each row)
    slow_steps = [
        ("Expanding contractions", pfunc.expand_contractions),
        ("Expanding slang", pfunc.expand_slang),
        ("Cleaning repeats", pfunc.clean_single_letters_and_repeats),
        ("Removing stopwords", pfunc.remove_stopwords),
    ]

    for name, step in slow_steps:
        print_info(f"→ {name}...", "blue")
        df["cleaned_text"] = df["cleaned_text"].progress_apply(step)

    # Save
    output_path = Path(config.dataset.preprocessed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print_info(f"✓ Saved to {output_path}", "green")


if __name__ == "__main__":
    main()