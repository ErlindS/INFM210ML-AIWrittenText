import kagglehub

# Download latest version
path = kagglehub.competition_download('llm-detect-ai-generated-text')

print("Path to competition files:", path)