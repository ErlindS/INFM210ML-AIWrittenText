import kagglehub
import shutil
from pathlib import Path

def downloadDaigtv2Dataset():
    path = kagglehub.dataset_download("thedrcat/daigt-v2-train-dataset")
    target = Path(__file__).resolve().parent / "daigt_v2"
    print("Target path for dataset files:", target)
    shutil.copytree(path, target, dirs_exist_ok=True)

def downloadLlmDetectAiGeneratedTextCompetition():
    # Download latest version
    path = kagglehub.competition_download('llm-detect-ai-generated-text')



if __name__ == "__main__":
    downloadDaigtv2Dataset()