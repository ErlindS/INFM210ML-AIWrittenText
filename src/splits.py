from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
DATA = Path(__file__).resolve().parent / "data"
MIN_ESSAYS = 1000  
SEED = 42

GENERATOR = {
   "chat_gpt_moth": "gpt", "radek_500": "gpt", "radekgpt4": "gpt",
   "llama2_chat": "llama2", "llama_70b_v1": "llama2", "NousResearch/Llama-2-7b-chat-hf": "llama2",
   "mistral7binstruct_v1": "mistral", "mistral7binstruct_v2": "mistral",
   "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
   "darragh_claude_v6": "claude", "darragh_claude_v7": "claude",
   "kingki19_palm": "palm", "palm-text-bison1": "palm",
   "falcon_180b_v1": "falcon",
   "cohere-command": "cohere",
}
df = pd.read_csv(DATA / "daigt_v2" / "train_v2_drcat_02.csv")
df["generator"] = df["source"].map(GENERATOR).where(df["label"] == 1, "human")
df = df.dropna(subset=["generator"]).reset_index(drop=True) 

# 1. Generatoren mit genug Essays auswählen
counts = df.loc[df["label"] == 1, "generator"].value_counts()
selected = counts[counts >= MIN_ESSAYS].index.tolist()
print("Essays pro Generator:\n", counts.to_string(), sep="")
print(f"\nLOGO-Folds (>= {MIN_ESSAYS} Essays): {selected}")

# 2. Random Split 
_, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["generator"], random_state=SEED)
df["random"] = "train"
df.loc[test_idx, "random"] = "test"

# 3. Folds:
human = df["generator"] == "human"
for g in selected:
   col = f"logo_{g}"
   df[col] = "train"
   df.loc[df["generator"] == g, col] = "test"
   df.loc[human, col] = df.loc[human, "random"]
split_cols = ["random"] + [f"logo_{g}" for g in selected]
summary = pd.DataFrame({c: df.groupby([c, "label"]).size() for c in split_cols}).T
summary.columns = [f"{part}_{'ki' if label else 'mensch'}" for part, label in summary.columns]
print("\nEssays pro Split:\n", summary.to_string(), sep="")
df.to_csv(DATA / "daigt_v2_splits.csv", index=False)
print(f"\nGespeichert: {DATA / 'daigt_v2_splits.csv'}")