# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import gradio as gr
from huggingface_hub import CommitScheduler, configure_http_backend, HfApi, HfFileSystem, create_repo, update_repo_settings
from PIL import Image
import typing as T
import shutil
import time
import asyncio
import threading
from io import BytesIO
import traceback

# 环境配置
load_dotenv()
HF_TOKEN: Optional[str] = os.getenv('HF_TOKEN')
HF_REPO_ID: str = os.getenv('HF_REPO_ID', 'default/dataset_repo')
GRADIO_USERNAME: Optional[str] = os.getenv("GRADIO_USERNAME")
GRADIO_PASSWORD: Optional[str] = os.getenv("GRADIO_PASSWORD")
API_BEARER_TOKEN: str = os.getenv("API_BEARER_TOKEN", "default_token")
HTTP_PROXY: Optional[str] = os.getenv('http_proxy')
HTTPS_PROXY: Optional[str] = os.getenv('https_proxy')

METADATA_FILENAME: str = 'metadata.jsonl'
LOCAL_STORAGE: str = 'image_dataset'
REPO_FOLDER_LIMIT: int = 10
SCHEDULER_EVERY: int = 5

# 常量
IMAGE_DATASET_DIR_PREFIX = "image_dataset"
MAX_IMAGES_PER_DIR = 10

# 初始化目录
Path(IMAGE_DATASET_DIR_PREFIX).mkdir(parents=False, exist_ok=True)






'''
1. 每次 push 前遍历目标文件系统, 检测文件是否存在
2. 
'''
class MyCommitScheduler(CommitScheduler):

    def update_folder(self, new_folder_path: str):
        if len(os.listdir(self.folder_path)) < MAX_IMAGES_PER_DIR:
            return
        # 文件满了, 等待同步线程退出
        self.stop()
        while self._scheduler_thread.is_alive():
            time.sleep(1)
        # 创建新的同步器
        self.__init__(
            repo_id         = self.repo_id,
            folder_path     = new_folder_path,
            every           = self.every,
            path_in_repo    = self.path_in_repo,
            repo_type       = self.repo_type,
            revision        = self.revision,
            token           = self.token,
            allow_patterns  = self.allow_patterns,
            ignore_patterns = self.ignore_patterns,
            squash_history  = self.squash_history,
            hf_api          = self.api,
        )
        pass

    def _push_to_hub(self) -> Dict[str, Any]:
        result = super()._push_to_hub()
        # Step 01. 检查已缓存的文件, 如果存在则删除, 排除元数据文件
        for path in self.last_uploaded:
            assert isinstance(path, Path)
            if path.exists() and METADATA_FILENAME != path.name:
                path.unlink()
        # Step 02. 尝试刷新目录
        self.update_folder()
        return result
    
    def stop(self):
        super().stop()
        return 


# 初始化 Scheduler
scheduler = MyCommitScheduler(
    repo_id="example-space-to-dataset-image",
    repo_type="dataset",
    folder_path=IMAGE_DATASET_DIR,
    path_in_repo=IMAGE_DATASET_DIR.name,
    every=5,
    squash_history=True,
    private=True,
)








def validate_environment_variables() -> None:
    """验证环境变量并显示警告（如果需要）"""
    if HF_TOKEN is None:
        print("Warning: HF_TOKEN environment variable is not set. You may encounter issues uploading to Hugging Face Hub.")
    if HF_REPO_ID == "your_username/your_dataset_repo_id":
        print("Warning: HF_REPO_ID is using the default value. Please set HF_REPO_ID environment variable.")
    if API_BEARER_TOKEN == "your_api_token":
        print("Warning: API_BEARER_TOKEN is using the default value. Please set API_BEARER_TOKEN environment variable.")


def configure_proxy() -> None:
    """为 Hugging Face Hub API 请求配置 HTTP 代理"""
    if HTTP_PROXY or HTTPS_PROXY:
        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.proxies = {
                "http": HTTP_PROXY,
                "https": HTTPS_PROXY,
            }
            return session
        configure_http_backend(backend_factory=backend_factory)


def save_image_to_dataset(filename: str, image_array: np.ndarray) -> None:
    """保存图片到数据集目录并更新元数据"""
    IMAGE_DATASET_DIR = Path()
    IMAGE_JSONL_PATH = Path()
    
    if not filename:
        filename = f"{uuid4()}.png"
    
    image_path = IMAGE_DATASET_DIR / f"{filename}"

    with scheduler.lock:
        Image.fromarray(image_array).save(image_path)
        with IMAGE_JSONL_PATH.open("a") as f:
            json.dump({
                "filename": filename,
                "file_path": image_path.name,
                "datetime": datetime.now().isoformat()
            }, f)
            f.write("\n")
    
    

def gradio_upload_image(image_file: Image.Image, filename_input: str) -> str:
    """处理来自 Gradio 界面的图片上传"""
    if image_file is None:
        return "没有上传图片"
    
    image_array = np.array(image_file)
    filename = filename_input if filename_input else f"unnamed_{uuid4()}.png"
    
    save_image_to_dataset(filename, image_array)
    
    return f"图片 '{filename}' 上传成功！（当前目录 {}/{} 张图片）"


def create_gradio_app() -> gr.Blocks:
    """创建并配置 Gradio 界面，包含正确的身份验证"""
    with gr.Blocks(title="图片上传到 Hugging Face 数据集") as demo:
        gr.Markdown("# 图片上传到 Hugging Face 数据集")
        gr.Markdown("上传图片到你的 Hugging Face 数据集存储库。")
        
        with gr.Row():
            image_input = gr.Image(type="pil", label="上传图片")
            filename_input = gr.Textbox(label="文件名（可选）", placeholder="在此输入文件名")
        
        upload_button = gr.Button("上传图片")
        result_output = gr.Textbox(label="结果")
        
        upload_button.click(
            fn=gradio_upload_image,
            inputs=[image_input, filename_input],
            outputs=result_output
        )
    
    return demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理应用程序生命周期事件"""

    IMAGE_DATASET_DIR = Path()
    
    if HF_TOKEN is None:
        print("Warning: HF_TOKEN is not set, commit scheduler will not start.")
    else:
        print(f"Commit scheduler started, saving files to {IMAGE_DATASET_DIR} and uploading to {HF_REPO_ID}")
        print(f"Current directory contains {}/{} images")
    
    yield
    
    print("Commit scheduler stopped.")


def init_app() -> FastAPI:
    """初始化并配置 FastAPI 应用程序"""
    application = FastAPI(lifespan=lifespan)
    
    # 创建带有正确配置的身份验证的 Gradio 应用
    gradio_auth = None
    if GRADIO_USERNAME and GRADIO_PASSWORD:
        gradio_auth = [(GRADIO_USERNAME, GRADIO_PASSWORD)]
        print(f"Gradio 身份验证已配置，用户名: {GRADIO_USERNAME}")
    else:
        print('警告: 未找到 Gradio 身份验证配置。')
    
    gradio_app = create_gradio_app()
    
    # 这是关键修复 - 在挂载应用时传递身份验证
    app_path = "/gradio"
    gr.mount_gradio_app(application, gradio_app, path=app_path, auth=gradio_auth)
    
    return application


# 初始化环境和服务
validate_environment_variables()
configure_proxy()
app = init_app()
security = HTTPBearer()


async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """验证 API 身份验证的 Bearer 令牌"""
    if credentials.scheme != "Bearer" or credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")
    return credentials.credentials


@app.post("/api/upload_image")
async def api_upload_image(
    file: UploadFile = File(...),
    filename: str = Form(...),
    api_key: str = Depends(get_api_key)
) -> Dict[str, str]:
    """上传图片的 API 端点"""
    try:
        image_array = np.array(Image.open(file.file))
        save_image_to_dataset(filename, image_array)
        return {"message": f"图片 '{filename}' 通过 API 成功上传！"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传图片失败: {str(e)}")


@app.get("/")
async def read_root() -> Dict[str, str]:
    """应用程序信息的根端点"""
    return {
        "message": "欢迎使用图片上传 API 和 Web UI！",
        "web_ui": "访问 /gradio 使用 Gradio Web UI。",
        "api_endpoint": "使用 POST /api/upload_image 通过 API 上传图片（需要 Bearer Token 身份验证）。"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
