
import kagglehub
from pathlib import Path
import shutil

# Download latest version
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")

# Convert to Path object
download_path = Path(path)

# Create "raw" directory in the same folder as this Python script
current_dir = Path(__file__).parent
raw_dir = current_dir / "raw"
raw_dir.mkdir(exist_ok=True)

# Copy dataset contents to "raw" folder
for item in download_path.iterdir():
    dest = raw_dir / item.name
    if item.is_dir():
        # Copy directory
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        # Copy file
        shutil.copy2(item, dest)

print(f"Dataset copied to: {raw_dir.resolve()}")