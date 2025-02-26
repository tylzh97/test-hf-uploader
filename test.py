
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
import json
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from huggingface_hub import CommitScheduler, configure_http_backend, HfApi, HfFileSystem, list_repo_tree, create_repo
from huggingface_hub.hf_api import RepoFolder, RepoFile
from PIL import Image

import typing as T
import time


# 环境配置
load_dotenv()
HF_TOKEN: Optional[str] = os.getenv('HF_TOKEN')
HF_REPO_ID: str = os.getenv('HF_REPO_ID')
METADATA_FILENAME: str = 'metadata.jsonl'
LOCAL_STORAGE: str = 'image_dataset'

REPO_FOLDER_LIMIT: int = 4
SCHEDULER_EVERY: int = 1


uploaded_buffer: List[str] = []

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
        self.__scheduler: T.Optional[CommitScheduler] = None
        self.initialize()

    def initialize(self):
        # Step 00. 获取工作目录, eg: '0000'
        self.repo_folder: str = self.get_current_repo_folder() 
        # Step 01. 检查远程目录结构, 完成本地 metadata 状态的同步
        self.initialize_local_dir()
        # Step 02. 创建 Scheduler
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
        for d in filter(lambda x:x['type'] == 'directory', self.fs.ls(self.repo_path, detail=True)):
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
            for f in filter(lambda x:x['type'] == 'file', self.fs.ls(f'{self.repo_path}/{last_dir}', detail=True)):
                files.append(f['name'])
            if len(files) >= REPO_FOLDER_LIMIT - 1:
                # 如果数量超过限制, 则创建一个新的子目录
                last_dir = self.get_next_available_folder(dirs)
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

    def check(self) -> int:
        """重新计算文件数目
        """
        with open(self.metadata_local_path, 'r') as f:
            lines: int = sum(1 for _ in f)
        if lines >= REPO_FOLDER_LIMIT - 1:
            self.switch_to_next()
            return self.recount()
        else:
            return lines

print(HF_REPO_ID)
sm = SchedulerManager(HF_REPO_ID)

for i in range(10):
    sm.check()
d = sm.switch_to_next()
print(d)

time.sleep(5)
print('Finish')
