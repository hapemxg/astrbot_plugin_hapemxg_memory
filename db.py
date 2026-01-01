# db.py

import json
import os
import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .models import MemoryAboutUser

class MemoryDB:
    def __init__(self):
        self.data: Dict[str, MemoryAboutUser] = {}
        self.buffer: Dict[str, List[str]] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.file_path = ""
        self.buffer_path = ""
        # 新增一个专门用于保存文件的锁
        self.save_lock = asyncio.Lock()

    def set_path(self, data_path: str):
        self.file_path = data_path
        self.buffer_path = data_path.replace("memory_data.json", "buffer_data.json")
        self.load()

    def load(self):
        # 加载记忆数据
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    for uid, mem_dict in raw_data.items():
                        self.data[uid] = MemoryAboutUser(**mem_dict)
            except Exception as e:
                from astrbot.api import logger
                logger.error(f"Error loading memory data: {e}")
        
        # 加载缓冲区和计数器数据
        if os.path.exists(self.buffer_path):
            try:
                with open(self.buffer_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # 兼容性处理：判断是新格式还是旧格式
                    if isinstance(raw, dict) and "buffers" in raw and "counters" in raw:
                        self.buffer = raw["buffers"]
                        # defaultdict 无法直接序列化还原，需手动转换
                        self.counters = defaultdict(int, raw["counters"])
                    else:
                        # 旧格式 (纯 buffer 字典)
                        self.buffer = raw
                        self.counters = defaultdict(int)
            except Exception as e:
                from astrbot.api import logger
                logger.error(f"Error loading buffer data: {e}")

    async def save_async(self):
        """异步保存数据和缓冲区"""
        # 使用新增的锁来确保文件操作的原子性
        async with self.save_lock:
            try:
                loop = asyncio.get_running_loop()
                raw_data = {uid: mem.model_dump() for uid, mem in self.data.items()}
                # 保存时将 defaultdict 转为 dict
                raw_buffer_file = {
                    "buffers": self.buffer,
                    "counters": dict(self.counters)
                }
                
                # 在线程池中执行文件写入
                await loop.run_in_executor(None, self._write_files, raw_data, raw_buffer_file)
            except Exception as e:
                from astrbot.api import logger
                # 使用 __class__.__name__ 获取当前类名，使日志更清晰
                logger.error(f"[{self.__class__.__name__}] Error saving data: {e}")

    def _write_files(self, data, buffer_file_content):
        # 写入记忆数据
        self._atomic_write(self.file_path, data)
        # 写入缓冲区数据
        self._atomic_write(self.buffer_path, buffer_file_content)

    def _atomic_write(self, path, content):
        if not path: return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, path)

    def get_memory(self, db_key: str) -> MemoryAboutUser:
        return self.data.get(db_key, MemoryAboutUser())

    def update_memory(self, db_key: str, memory: MemoryAboutUser):
        self.data[db_key] = memory
        asyncio.create_task(self.save_async())

    def get_buffer(self, db_key: str) -> List[str]:
        return self.buffer.get(db_key, [])

    def add_to_buffer(self, db_key: str, message: str, max_len: Optional[int] = None):
        """
        添加消息到缓冲区
        :param max_len: 滑动窗口大小
        """
        if db_key not in self.buffer:
            self.buffer[db_key] = []
        
        self.buffer[db_key].append(message)
        
        # 滑动窗口逻辑
        if max_len is not None and max_len > 0:
            current_len = len(self.buffer[db_key])
            if current_len > max_len:
                self.buffer[db_key] = self.buffer[db_key][-max_len:]
        
        asyncio.create_task(self.save_async())

    def clear_buffer(self, db_key: str):
        if db_key in self.buffer:
            self.buffer[db_key] = []
            asyncio.create_task(self.save_async())

    # --- 计数器相关方法 ---
    
    def increment_counter(self, db_key: str):
        self.counters[db_key] += 1
        asyncio.create_task(self.save_async())

    def get_counter(self, db_key: str) -> int:
        return self.counters[db_key]

    def reset_counter(self, db_key: str):
        self.counters[db_key] = 0
        asyncio.create_task(self.save_async())

db = MemoryDB()