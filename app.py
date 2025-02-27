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
import aiofiles
import aiofiles.os
import logging
from pydantic import BaseSettings

load_dotenv()

# 配置
class Settings(BaseSettings):
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
    SCHEDULER_EVERY: int = 3

    IMAGE_DATASET_DIR_PREFIX = "image_dataset"
    MAX_IMAGES_PER_DIR = 10

settings = Settings()

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

uploaded_buffer: List[str] = []  # 缓存上传的文件路径，提交后清理

# 初始化目录
Path(settings.IMAGE_DATASET_DIR_PREFIX).mkdir(parents=False, exist_ok=True)

class MyCommitScheduler(CommitScheduler):
    """自定义 CommitScheduler, 在推送成功后清理本地文件"""
    
    def _push_to_hub(self) -> Dict[str, Any]:
        """重写推送方法，在推送成功后清理文件"""
        global uploaded_buffer
        logging.info(f'正在执行推送任务...{id(self)}')
        try:
            result = super()._push_to_hub()
            for file in uploaded_buffer:
                file_path = Path(file)
                if file_path.exists():
                    file_path.unlink()
            uploaded_buffer = []
            logging.info(f'推送任务执行成功')
            return result
        except Exception as e:
            logging.exception(f"推送任务执行失败: {e}")
            raise e
        finally:
            logging.info(f'推送任务执行完成')
    
    def stop(self):
        logging.info(f'Before MyCommitScheduler.stop')
        super().stop()
        logging.info(f'After MyCommitScheduler.stop')
        return


class SchedulerManager:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.repo_path = f'datasets/{self.repo_id}'
        self.hf_api = HfApi()
        self.fs = HfFileSystem()
        self.__scheduler: T.Optional[CommitScheduler] = None
        self.initialize()

    def initialize(self):
        # Step 00. 获取工作目录, eg: '0000'
        self.repo_folder: str = self.get_current_repo_folder() 
        logging.info(f'新的仓库存储目录: {self.repo_folder}')
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
                every=settings.SCHEDULER_EVERY,
                squash_history=True,
                private=True,
            )
        return self.__scheduler

    def initialize_local_dir(self):
        repo_folder: str = self.repo_folder
        self.local_folder: str = f'{settings.LOCAL_STORAGE}/{repo_folder}'
        local_folder = self.local_folder
        if not os.path.exists(local_folder):
            os.makedirs(local_folder, exist_ok=False)
        self.metadata_local_path: str = f'{local_folder}/{settings.METADATA_FILENAME}'
        metadata_repo_path: str = f'{self.repo_path}/{repo_folder}/{settings.METADATA_FILENAME}'

        # Step 01. 检查 metadata 是否存在, 如果存在则同步
        if self.fs.exists(metadata_repo_path):
            logging.info('从远程获取 metadata')
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
        # 由于缓存问题, 此处不能够使用 HfFileSystem 接口
        if not self.hf_api.repo_exists(self.repo_id, repo_type='dataset'):
            logging.info(f'创建了一个新的仓库')
            create_repo(self.repo_id, repo_type='dataset', private=True, exist_ok=False)
            while self.hf_api.repo_exists(self.repo_id, repo_type='dataset') is False:
                time.sleep(5)
      
        # Step 02. 检查远程仓库中的目录结构
        dirs: T.List[str] = []
        logging.info(self.repo_path)
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
            if len(files) >= REPO_FOLDER_LIMIT - 1:
                # 如果数量超过限制, 则创建一个新的子目录
                last_dir = self.get_next_available_folder(dirs)
        else:
            # Step 04. 如果不存在, 则创建一个新的子目录, 编号从 0 开始
            last_dir = self.get_next_available_folder(dirs)
        return last_dir

    def stop(self):
        async def sync_and_remove(scheduler: CommitScheduler, folder: str, retry: int=0):
            if retry > 3:
                logging.error(f'重试多次后依然失败, 已经结束...')
                return
            success: bool = False
            try:
                x = BytesIO()
                with scheduler as s:
                    x.write(str(s).encode())
                del scheduler
                success = True
            except Exception as e:
                logging.exception(f'同步旧的调度器时出现了问题: {e}')
                traceback.print_exc()
            finally:
                logging.info(f'成功同步并清理目录: {folder}')
            if success:
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    logging.error(f"删除目录 {folder} 失败: {e}")
                return
            else:
                await asyncio.sleep(60)
                await sync_and_remove(scheduler, folder, retry+1)
        
        # Step 01. 在子线程中执行旧的同步任务
        if self.__scheduler:  # 确保 scheduler 存在
            asyncio.create_task(sync_and_remove(self.__scheduler, self.local_folder))
            self.__scheduler = None
        return

    def switch_to_next(self) -> str:
        """切换到下一个目录
        """
        self.stop()
        # 强制从远端获取一个新的目录
        self.initialize()
        return self.repo_folder

    def count(self) -> int:
        try:
            with open(self.metadata_local_path, 'r') as f:
                lines: int = sum(1 for _ in f if _.strip())
            return lines
        except FileNotFoundError:
            return 0

    def check(self) -> str:
        lines = self.count()
        if lines >= settings.MAX_IMAGES_PER_DIR - 1:
            self.switch_to_next()
            return f"切换到新目录: {self.repo_folder} ({self.count()}/{settings.MAX_IMAGES_PER_DIR})"
        else:
            return f"当前目录: {self.repo_folder} ({self.count()}/{settings.MAX_IMAGES_PER_DIR})"

def validate_environment_variables() -> None:
    if settings.HF_TOKEN is None:
        logging.warning("Warning: HF_TOKEN environment variable is not set. You may encounter issues uploading to Hugging Face Hub.")
    if settings.HF_REPO_ID == "your_username/your_dataset_repo_id":
        logging.warning("Warning: HF_REPO_ID is using the default value. Please set HF_REPO_ID environment variable.")
    if settings.API_BEARER_TOKEN == "your_api_token":
        logging.warning("Warning: API_BEARER_TOKEN is using the default value. Please set API_BEARER_TOKEN environment variable.")

    if not settings.HF_REPO_ID:
        raise ValueError("HF_REPO_ID is required.")
    if not settings.API_BEARER_TOKEN:
        raise ValueError("API_BEARER_TOKEN is required.")

def configure_proxy() -> None:
    """为 Hugging Face Hub API 请求配置 HTTP 代理"""
    if HTTP_PROXY or HTTPS_PROXY:
        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.proxies = {
                "http": settings.HTTP_PROXY,
                "https": settings.HTTPS_PROXY,
            }
            return session
        configure_http_backend(backend_factory=backend_factory)

async def save_image_to_dataset(filename: str, image_array: np.ndarray, scheduler_manager: SchedulerManager) -> None:
    """保存图片到数据集目录并更新元数据"""
    scheduler = scheduler_manager.get_scheduler()
    IMAGE_DATASET_DIR = Path(scheduler_manager.local_folder)
    IMAGE_JSONL_PATH = Path(scheduler_manager.metadata_local_path)

    current_dir_status = scheduler_manager.check()

    if not filename:
        filename = f"{uuid4()}.png"

    image_path = IMAGE_DATASET_DIR / f"{filename}"

    try:
        with scheduler.lock:
            image = Image.fromarray(image_array)
            image.save(image_path)
            async with aiofiles.open(IMAGE_JSONL_PATH, "a") as f:
                await f.write(json.dumps({
                    "filename": filename,
                    "file_path": image_path.name,
                    "datetime": datetime.now().isoformat()
                }) + "\n")
            uploaded_buffer.append(str(image_path))

        logging.info(f"图片 '{filename}' 保存到数据集目录: {image_path}")
    except Exception as e:
        logging.exception(f"保存图片 '{filename}' 失败: {e}")
        raise

async def gradio_upload_image(
    image_file: Image.Image,
    filename_input: str,
    scheduler_manager: SchedulerManager = Depends(lambda: scheduler_manager)
) -> str:
    """处理来自 Gradio 界面的图片上传"""
    if image_file is None:
        return "没有上传图片"

    image_array = np.array(image_file)
    filename = filename_input if filename_input else f"unnamed_{uuid4()}.png"

    try:
        await save_image_to_dataset(filename, image_array, scheduler_manager)
        return f"图片 '{filename}' 上传成功！（{scheduler_manager.check()}）"
    except Exception as e:
        logging.error(f"图片上传失败: {e}")
        return f"图片上传失败: {e}"

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
    try:
        validate_environment_variables()
        configure_proxy()

        global scheduler_manager
        scheduler_manager = SchedulerManager(settings.HF_REPO_ID)

        if settings.HF_TOKEN is None:
            logging.warning("Warning: HF_TOKEN is not set, commit scheduler will not start.")
        else:
            logging.info(f"Commit scheduler started, saving files to {scheduler_manager.local_folder} and uploading to {settings.HF_REPO_ID}")
            logging.info(f"{scheduler_manager.check()}") # 打印初始化状态

        yield
    finally:
        if scheduler_manager:
            scheduler_manager.stop() # 在应用关闭时停止 scheduler
        logging.info("Application shutdown complete.")

def init_app() -> FastAPI:
    """初始化并配置 FastAPI 应用程序"""
    application = FastAPI(lifespan=lifespan)

    gradio_auth = None
    if settings.GRADIO_USERNAME and settings.GRADIO_PASSWORD:
        gradio_auth = [(settings.GRADIO_USERNAME, settings.GRADIO_PASSWORD)]
        logging.info(f"Gradio 身份验证已配置，用户名: {settings.GRADIO_USERNAME}")
    else:
        logging.warning('警告: 未找到 Gradio 身份验证配置。')

    gradio_app = create_gradio_app()

    app_path = "/gradio"
    gr.mount_gradio_app(application, gradio_app, path=app_path, auth=gradio_auth)

    return application

app = init_app()
security = HTTPBearer()

async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """验证 API 身份验证的 Bearer 令牌"""
    if credentials.scheme != "Bearer" or credentials.credentials != settings.API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="无效的 API 密钥")
    return credentials.credentials

@app.post("/api/upload_image")
async def api_upload_image(
    file: UploadFile = File(...),
    filename: str = Form(...),
    api_key: str = Depends(get_api_key),
    scheduler_manager: SchedulerManager = Depends(lambda: scheduler_manager)
) -> Dict[str, str]:
    """上传图片的 API 端点"""
    try:
        image_array = np.array(Image.open(file.file))
        await save_image_to_dataset(filename, image_array, scheduler_manager)
        return {"message": f"图片 '{filename}' 通过 API 成功上传！"}
    except Exception as e:
        logging.error(f"API 上传图片失败: {e}")
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
