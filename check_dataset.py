import json

# Function to load .jsonl
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load data
train_data = load_jsonl("train.jsonl")
val_data = load_jsonl("val.jsonl")
test_data = load_jsonl("test.jsonl")

# Print first 3 train samples
for i, sample in enumerate(train_data[:3]):
    cluster_id = sample['cluster_id']
    articles = sample['articles']
    summary = sample['summary']

    print(f"\nSample {i+1}:")
    print("Cluster ID:", cluster_id)
    print("Number of articles in cluster:", len(articles))
    print("First article snippet:", articles[0][:200], "...")
    print("Summary snippet:", summary[:200], "...")
