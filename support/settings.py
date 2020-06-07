from pathlib import Path
import nltk

p = Path("trained")
if p.is_dir():
    print("directory for trained already exists.")
else:
    Path("trained").mkdir()
    print("directory for trained is created!")

nltk.download()
