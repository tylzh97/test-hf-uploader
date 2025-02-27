
# https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi
repo_id = "megatrump/Space2Dataset"
api = HfApi()

load_dotenv()

HF_TOKEN: str           = os.environ.get('HF_TOKEN', '')
HF_REPO_ID: str         = os.environ.get('HF_REPO_ID', '')
API_BEARER_TOKEN: str   = os.environ.get('API_BEARER_TOKEN', '')
GRADIO_USERNAME: str    = os.environ.get('GRADIO_USERNAME', '')
GRADIO_PASSWORD: str    = os.environ.get('GRADIO_PASSWORD', '')

# 添加 Secrets
api.add_space_secret(repo_id=repo_id, key="HF_TOKEN",           value=HF_TOKEN)
api.add_space_secret(repo_id=repo_id, key="HF_REPO_ID",         value=HF_REPO_ID)
api.add_space_secret(repo_id=repo_id, key="API_BEARER_TOKEN",   value=API_BEARER_TOKEN)
api.add_space_secret(repo_id=repo_id, key="GRADIO_USERNAME",    value=GRADIO_USERNAME)
api.add_space_secret(repo_id=repo_id, key="GRADIO_PASSWORD",    value=GRADIO_PASSWORD)
