from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="bartowski/TinySwallow-1.5B-Instruct-GGUF",
    filename="TinySwallow-1.5B-Instruct-Q4_K_L.gguf",
    local_dir="./models",
    local_dir_use_symlinks=False
)
print(f"Model downloaded to: {path}")
