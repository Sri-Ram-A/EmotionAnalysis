import kagglehub, shutil, os
path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")
print("Original path:", path)
desired = r"raw"
os.makedirs(desired, exist_ok=True)
shutil.move(path, desired)  # or shutil.copytree if it's a dir
print("Moved to:", desired)
