from hugginface_hub import snapshot_download

model_id = "meta-llama/Llama-2-7b-hf"
snapshot_download(repo_id = model_id , local_dir = r"/llama_7b_hf",local_dir_use_symlinks = False,revision = "main")