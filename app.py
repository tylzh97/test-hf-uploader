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
from huggingface_hub import CommitScheduler, configure_http_backend, HfApi, HfFileSystem, create_repo
from PIL import Image
import typing as T
import shutil
import time
import asyncio


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
REPO_FOLDER_LIMIT: int = 4
SCHEDULER_EVERY: int = 1

# 常量
IMAGE_DATASET_DIR_PREFIX = "image_dataset"
MAX_IMAGES_PER_DIR = 10
uploaded_buffer: List[str] = []  # 缓存上传的文件路径，提交后清理


class MyCommitScheduler(CommitScheduler):
    """自定义 CommitScheduler, 在推送成功后清理本地文件"""
    
    def _push_to_hub(self) -> Dict[str, Any]:
        """重写推送方法，在推送成功后清理文件"""
        global uploaded_buffer
        try:
            result = super()._push_to_hub()
            for file in uploaded_buffer:
                file_path = Path(file)
                if file_path.exists():
                    file_path.unlink()
            uploaded_buffer = []
            return result
        except Exception as e:
            raise e
    
    def stop(self):
        print(f'Before MyCommitScheduler.stop')
        super().stop()
        print(f'After MyCommitScheduler.stop')
        return 


class SchedulerManager:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.repo_path = f'datasets/{self.repo_id}'
        self.hf_api = HfApi()
        self.fs = HfFileSystem()
        self.force: bool = False
        self.__scheduler: T.Optional[CommitScheduler] = None
        self.initialize()

    def initialize(self):
        # Step 00. 获取工作目录, eg: '0000'
        self.repo_folder: str = self.get_current_repo_folder() 
        print(f'新的仓库存储目录: {self.repo_folder}')
        # Step 01. 检查远程目录结构, 完成本地 metadata 状态的同步
        self.initialize_local_dir()
        # Step 02. 创建 Scheduler
        self.__scheduler = None
        self.__scheduler = self.get_scheduler()
        return

    def get_scheduler(self) -> CommitScheduler:
        if self.__scheduler is None:
            self.__scheduler = MyCommitScheduler(
                repo_id=self.repo_id,
                repo_type='dataset',
                folder_path=self.local_folder,
                path_in_repo=self.repo_folder,
                every=SCHEDULER_EVERY,
                squash_history=True,
                private=True,
            )
        return self.__scheduler

    def initialize_local_dir(self):
        """获取应当存储的远程目录, 并在本地目录中同步 metadata
        """
        repo_folder: str = self.repo_folder
        self.local_folder: str = f'{LOCAL_STORAGE}/{repo_folder}'
        local_folder = self.local_folder
        if not os.path.exists(local_folder):
            os.mkdir(local_folder)
        self.metadata_local_path: str = f'{local_folder}/{METADATA_FILENAME}'
        metadata_repo_path: str = f'{self.repo_path}/{repo_folder}/{METADATA_FILENAME}'

        # Step 01. 检查 metadata 是否存在, 如果存在则同步
        if self.fs.exists(metadata_repo_path):
            print('从远程获取 metadata')
            self.fs.download(metadata_repo_path, self.metadata_local_path)
        else:
            with open(self.metadata_local_path, 'w') as f:
                f.write('')
        return

    def get_next_available_folder(self, current: T.List[str]) -> str:
        return sorted(set(f'{_:04d}' for _ in range(1000)) - set(current))[0]

    def get_current_repo_folder(self) -> Path:
        """获取远程仓库中应该存储的路径

        该函数有调用成本, 因此应当仅以下情况调用:
            - 初始化仓库时
            - 本地目录中存储满时
        """
        # Step 01. 检查远程仓库是否存在, 如果不存在则创建一个新的仓库
        if not self.fs.exists(self.repo_path):
            create_repo(self.repo_id, repo_type='dataset', private=True)
      
        # Step 02. 检查远程仓库中的目录结构
        dirs: T.List[str] = []
        for d in filter(lambda x:x['type'] == 'directory', self.fs.ls(self.repo_path, detail=True, refresh=True)):
            '''
            {
                'name': 'datasets/megatrump/test-img2/img_000', 
                'size': 0, 
                'type': 
                'directory', 
                'tree_id': 'e267c85605ba3c1f743f68d72c0bdd1a08fd5bf7', 
                'last_commit': LastCommitInfo(
                    oid='f10e98e1f1b1914d281cc72e3078eed59235682a', 
                    title="Super-squash branch 'main' using huggingface_hub", 
                    date=datetime.datetime(2025, 2, 26, 10, 27, 56, tzinfo=datetime.timezone.utc)
                )
            }
            '''
            name = d['name']
            if name.startswith(self.repo_path):
                name = name[len(self.repo_path):]
            name = name.strip('/')
            dirs.append(name)
        
        # Step 03. 检查编号最大的子目录中, 是否存储 '满' 了
        if dirs:
            last_dir: str = sorted(dirs)[-1]
            # 则检查子目录中的文件数量
            files: T.List[str] = []
            for f in filter(lambda x:x['type'] == 'file', self.fs.ls(f'{self.repo_path}/{last_dir}', detail=True, refresh=True)):
                files.append(f['name'])
            if len(files) >= REPO_FOLDER_LIMIT - 1 or self.force:
                # 如果数量超过限制, 则创建一个新的子目录
                print(dirs)
                print(files)
                last_dir = self.get_next_available_folder(dirs)
                print(last_dir)
        else:
            # Step 04. 如果不存在, 则创建一个新的子目录, 编号从 0 开始
            last_dir = self.get_next_available_folder(dirs)
        return last_dir

    def stop(self):
        # Step 01. 同步当前的目录
        with self.__scheduler as s:
            # 通过 CommitScheduler.__exit__, 强制触发一次目录同步
            self.__scheduler = None
        shutil.rmtree(self.local_folder)

    def switch_to_next(self) -> str:
        """切换到下一个目录
        """
        self.stop()
        # 强制从远端获取一个新的目录
        self.initialize()
        return self.repo_folder

    def count(self) -> int:
        with open(self.metadata_local_path, 'r') as f:
            lines: int = sum(1 for _ in f if _.strip())
        return lines

    def check(self) -> int:
        """重新计算文件数目
        """
        lines = self.count()
        if lines >= REPO_FOLDER_LIMIT - 1:
            # self.force = True
            self.switch_to_next()
            return self.count()
        else:
            return lines


# 全局变量
scheduler_manager: Optional['SchedulerManager'] = SchedulerManager(HF_REPO_ID)



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
    global scheduler_manager
    
    scheduler = scheduler_manager.get_scheduler()
    IMAGE_DATASET_DIR = Path(scheduler_manager.local_folder)
    IMAGE_JSONL_PATH = Path(scheduler_manager.metadata_local_path)
    # 首先检查是否需要切换目录（对于之前已满的目录）
    scheduler_manager.check()
    
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
    
    uploaded_buffer.append(str(image_path))
    

def gradio_upload_image(image_file: Image.Image, filename_input: str) -> str:
    """处理来自 Gradio 界面的图片上传"""
    global scheduler_manager
    if image_file is None:
        return "没有上传图片"
    
    image_array = np.array(image_file)
    filename = filename_input if filename_input else f"unnamed_{uuid4()}.png"
    
    save_image_to_dataset(filename, image_array)
    
    return f"图片 '{filename}' 上传成功！（当前目录 {scheduler_manager.check()}/{MAX_IMAGES_PER_DIR} 张图片）"


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
    global scheduler_manager

    CURRENT_IMAGES_COUNT = scheduler_manager.check()
    IMAGE_DATASET_DIR = Path(scheduler_manager.local_folder)
    
    if HF_TOKEN is None:
        print("Warning: HF_TOKEN is not set, commit scheduler will not start.")
    else:
        print(f"Commit scheduler started, saving files to {IMAGE_DATASET_DIR} and uploading to {HF_REPO_ID}")
        print(f"Current directory contains {CURRENT_IMAGES_COUNT}/{MAX_IMAGES_PER_DIR} images")
    
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
