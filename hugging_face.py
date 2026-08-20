from huggingface_hub import create_repo

# Create a private or public dataset repository
create_repo(repo_id="your-username/beir-precomputed-embeddings", repo_type="dataset")