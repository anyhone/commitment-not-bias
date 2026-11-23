"""Downloads the datasets into data/.

- BBQ: the jsonl files of the official repository (github.com/nyu-mll/BBQ), checked against the
  SHA-256 hashes of the copy used for the reported results (downloaded from the `main` branch on 23/10/2026).
- StereoSet (validation) and MMLU: from the Hugging Face Hub, saved as parquet.
"""
import hashlib
import urllib.request
from pathlib import Path

from datasets import load_dataset

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

BBQ_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{}.jsonl"
BBQ_SHA256 = {
    "Age": "46e805b3fc2d8cbd26eeb8e8430d98cf7b2dc9c83574ff3674e8ce4f0fca2a60",
    "Disability_status": "375d40d9b71f056150264445ec189b13934998f3f49a1e311aa7578eb3b45e5f",
    "Gender_identity": "8b5adbb368510a97d5775dc43e1c719e0249fe2da33f13c83e3e29edafec7a00",
    "Nationality": "a583e74666aec0341ded8aa3983e7cad694eb24de72c254d28b2df3035720090",
    "Physical_appearance": "e48d7e13508565f4801574e611043470c25c95a0a62af3699cb5ee7cb02608fb",
    "Race_ethnicity": "4a9f1214cfaa115ce7f0bdb40609122e431f5152c71b8ca64d665e483b946ae6",
    "Religion": "cb9555f9f3454a52cd2df85956b59bd7fcca5aa922c8527693b1164d08417616",
    "SES": "9f92754bb037b0982604b9112705fb81d60a19d9e759c67e4a85e484a070f528",
    "Sexual_orientation": "2c71036b9e7584fe589c42aef32c1a42bc01b9a5e9b1b8704342630cdb08cefd",
}

bbq_dir = DATA / "bbq"
bbq_dir.mkdir(exist_ok=True)
for cat, sha in BBQ_SHA256.items():
    path = bbq_dir / f"{cat}.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(BBQ_URL.format(cat), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha:
        raise SystemExit(f"BBQ {cat}: SHA-256 {digest} differs from the version used for the reported results ({sha})")
    print("bbq", cat, "ok")

for cfg in ["intrasentence", "intersentence"]:
    ds = load_dataset("McGill-NLP/stereoset", cfg, split="validation")
    ds.to_parquet(DATA / f"stereoset_{cfg}.parquet")
    print("stereoset", cfg, len(ds))

for split in ["test", "validation", "dev", "auxiliary_train"]:
    name = "auxiliary_train" if split == "auxiliary_train" else "all"
    ds = load_dataset("cais/mmlu", name, split="train" if split == "auxiliary_train" else split)
    if split == "auxiliary_train" and "train" in ds.column_names:  # nested format of cais/mmlu
        ds = ds.flatten()
    ds.to_parquet(DATA / f"mmlu_{split}.parquet")
    print("mmlu", split, len(ds), ds.column_names)
