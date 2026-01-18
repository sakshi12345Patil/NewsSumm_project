import pandas as pd
import json
import os

INPUT_FILE = "NewsSumm Dataset.xlsx"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading file...")

try:
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
except Exception:
    print("openpyxl failed, trying latin encoding...")
    df = pd.read_csv(INPUT_FILE, encoding="latin1")

print("Columns found:", df.columns)

# CHANGE column names if needed
ARTICLE_COL = "articles"
SUMMARY_COL = "summary"

samples = []

for _, row in df.iterrows():
    if pd.isna(row[ARTICLE_COL]) or pd.isna(row[SUMMARY_COL]):
        continue

    sample = {
        "articles": str(row[ARTICLE_COL]).split("|||"),
        "summary": str(row[SUMMARY_COL])
    }
    samples.append(sample)

print(f"Total samples: {len(samples)}")

# Split dataset
train = samples[:int(0.8 * len(samples))]
val = samples[int(0.8 * len(samples)):int(0.9 * len(samples))]
test = samples[int(0.9 * len(samples)):]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train, f"{OUTPUT_DIR}/train.jsonl")
save_jsonl(val, f"{OUTPUT_DIR}/val.jsonl")
save_jsonl(test, f"{OUTPUT_DIR}/test.jsonl")

print("Conversion done ✅")
