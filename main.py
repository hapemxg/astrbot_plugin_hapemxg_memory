# main.py (修正版)

import json
import asyncio
import re
from collections import defaultdict
from typing import List, Dict, Optional

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from astrbot.api.provider import ProviderRequest, LLMResponse
# 确保以下引用存在
import astrbot.api.message_components as Comp
from astrbot.api.message_components import BaseMessageComponent

from .db import db
from .models import MemoryAboutUser, MemoryResult

@register("hapemxg_memory", "hapemxg", "给AstrBot添加长久记忆功能", "0.1.3")
class HapeMemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        data_dir = StarTools.get_data_dir("hapemxg_memory")
        db.set_path(str(data_dir / "memory_data.json"))
        
        self.HISTORY_WINDOW_SIZE = self.config.get("history_window_size", 30)
        self.UPDATE_INTERVAL = self.config.get("update_interval", 10)
        
        self.clean_patterns = self.config.get("clean_patterns", [])
        self.lock = asyncio.Lock()

    def _get_db_key(self, event: AstrMessageEvent) -> str:
        return self._resolve_db_key(event, None)

    # main.py (修正后的代码)
    def _resolve_db_key(self, event: AstrMessageEvent, target_user_id: Optional[str]) -> str:
        final_user_id = target_user_id if target_user_id else event.get_sender_id()
        if self.config.get("memory_scope", False):
            # 从 event.message_obj 中获取 group_id
            group_id = event.message_obj.group_id
            # 如果 group_id 存在 (不是空字符串), 则说明是群聊
            if group_id:
                return f"{group_id}_{final_user_id}"
        # 如果关闭了会话隔离, 或者是私聊 (group_id为空), 则直接返回用户ID
        return final_user_id

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        for pattern in self.clean_patterns:
            try:
                # 使用 re.escape 来处理特殊字符，但这会破坏用户的本意，因此注释掉
                # 更建议用户编写正确的正则表达式，例如用 \[ 和 \] 替代 [ 和 ]
                cleaned = re.sub(pattern, "", cleaned)
            except re.error as e:
                logger.error(f"[Memory] Invalid regex pattern '{pattern}': {e}")
        return cleaned.strip()

    def _message_chain_to_text(self, chain: List[BaseMessageComponent]) -> str:
        """[修正] 将消息链稳健地转换为字符串，用占位符代表图片、At等非文本内容。"""
        if not chain:
            return ""

        text_parts = []
        for component in chain:
            if isinstance(component, Comp.Plain):
                if component.text and component.text.strip():
                    text_parts.append(component.text.strip())
            elif isinstance(component, Comp.Image):
                text_parts.append("[用户发送了一张图片]")
            elif isinstance(component, Comp.At):
                text_parts.append(f"[@{getattr(component, 'qq', '未知用户')}]")
        
        # --- 核心修改在这里 ---
        # 使用列表推导式替换 filter() 来避免命名冲突
        full_text = " ".join([part for part in text_parts if part]).strip()
        
        return self._clean_text(full_text)

    @filter.command_group("memory")
    def memory_group(self):
        pass

    @memory_group.command("status")
    async def memory_status(self, event: AstrMessageEvent, user_id: str = None):
        """查看当前记忆和更新进度
        
        参数:
            user_id (可选): 指定查询的用户ID。如果不填则查询自己。查询他人需要管理员权限。
        """
        sender_id = event.get_sender_id()
        
        # --- 新增逻辑: 权限检查 (修复版) ---
        if user_id and user_id != sender_id:
            # [修复] 尝试从 Context 的内部属性 _config 中获取全局管理员列表
            # 这里的 getattr 是为了兼容性，优先尝试 _config
            global_config = getattr(self.context, "_config", {})
            if not global_config:
                 # 再次尝试 config (防止未来版本改回)
                 global_config = getattr(self.context, "config", {})
            
            # 获取 admins_id，默认为空列表
            admins = global_config.get("admins_id", [])
            
            if sender_id not in admins:
                yield event.plain_result("只有管理员可以查看其他用户的记忆状态。")
                return

        # --- 键值解析 ---
        db_key = self._resolve_db_key(event, user_id)
        
        # --- 原有逻辑 ---
        memory = db.get_memory(db_key)
        counter = db.get_counter(db_key)
        buffer_len = len(db.get_buffer(db_key))
        
        mem_text = memory.to_text()
        
        info = f"【当前状态】\n缓冲队列: {buffer_len}/{self.HISTORY_WINDOW_SIZE} 条\n更新进度: {counter}/{self.UPDATE_INTERVAL} (再过 {max(0, self.UPDATE_INTERVAL - counter)} 条消息触发更新)\n\n"
        
        target_name = f"用户 {user_id}" if user_id and user_id != sender_id else "你"
        
        if not mem_text.strip():
             yield event.plain_result(info + f"我对{target_name}还没有什么印象呢~ 多聊聊吧！")
        else:
            yield event.plain_result(info + f"我对{target_name}的印象：\n{mem_text}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @memory_group.command("forget")
    async def memory_forget(self, event: AstrMessageEvent, user_id: str = None):
        """[管理员] 遗忘记忆"""
        db_key = self._resolve_db_key(event, user_id)
        
        db.update_memory(db_key, MemoryAboutUser())
        async with self.lock:
            db.clear_buffer(db_key)
            db.reset_counter(db_key)
            
        target_msg = f"用户 {user_id}" if user_id else "你"
        yield event.plain_result(f"已将关于 {target_msg} 的一切遗忘... 是新的开始呢。")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @memory_group.command("update")
    async def memory_force_update(self, event: AstrMessageEvent, user_id: str = None):
        """[管理员] 强制触发更新"""
        db_key = self._resolve_db_key(event, user_id)
        
        async with self.lock:
            buffer_copy = db.get_buffer(db_key)
        
        target_msg = f"用户 {user_id}" if user_id else "你"

        if not buffer_copy:
            yield event.plain_result(f"最近没有关于 {target_msg} 的新对话记录，无法更新记忆。")
            return
            
        yield event.plain_result(f"正在尝试根据最近的对话更新 {target_msg} 的记忆...")
        
        db.reset_counter(db_key)
        asyncio.create_task(self._update_memory_task(db_key, buffer_copy))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @memory_group.command("set")
    async def memory_set(self, event: AstrMessageEvent, key: str, value: str, user_id: str = None):
        """[管理员] 手动修改记忆"""
        db_key = self._resolve_db_key(event, user_id)
        mem = db.get_memory(db_key)
        
        if hasattr(mem, key):
            setattr(mem, key, value)
            db.update_memory(db_key, mem)
            target_msg = f"用户 {user_id}" if user_id else "自己"
            yield event.plain_result(f"已更新 {target_msg} 的 {key} 为: {value}")
        else:
            yield event.plain_result(f"找不到字段: {key}。可选字段：disposition, interests, doings, works, wishes, worries, skills, attitudes_to_you, experiences_with_you, extra_info")

    @filter.llm_tool(name="update_user_memory")
    async def update_user_memory(self, event: AstrMessageEvent, reason: str):
        """主动触发记忆更新"""
        call_count = getattr(event, '_memory_tool_call_count', 0)
        call_count += 1
        setattr(event, '_memory_tool_call_count', call_count)

        if call_count > 1:
            return json.dumps({"success": False, "message": "记忆更新任务已在本次对话中触发过，无需重复调用。"}, ensure_ascii=False)

        db_key = self._get_db_key(event)
        
        async with self.lock:
            buffer_copy = db.get_buffer(db_key)
            
        if not buffer_copy:
            return json.dumps({"success": False, "message": "最近没有新的对话记录，无法提取记忆。"}, ensure_ascii=False)
            
        db.reset_counter(db_key)
        asyncio.create_task(self._update_memory_task(db_key, buffer_copy, reason=reason))
        
        return json.dumps({
            "success": True, 
            "message": "记忆更新任务已成功在后台启动。"
        }, ensure_ascii=False)

    @filter.on_llm_request(priority=-1)
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        """请求前：注入记忆 + 记录用户消息 + 增加计数"""
        db_key = self._get_db_key(event)
        if not db_key:
            return

        # 1. 注入记忆和会话信息
        memory = db.get_memory(db_key)
        mem_text = memory.to_text()
        
        group_id = getattr(event, "group_id", "私聊")
        sender_id = event.get_sender_id()

        memory_prompt = (
            f"\n\n--- [用户记忆系统] ---\n"
            f"当前群号: {group_id}\n"
            f"当前用户QQ号: {sender_id}\n"
            f"你有一些关于该用户的记忆：\n{mem_text if mem_text else '尚无记忆。'}\n\n"
            f"[提示] 如果你发现了关于用户的新的重要信息，你可以使用 'update_user_memory' 工具主动更新用户画像。\n"
            f"--- [用户记忆结束] ---\n"
        )

        if req.system_prompt is None:
            req.system_prompt = ""
        req.system_prompt += memory_prompt
        
        if self.config.get("debug_memory"):
            log_message = (
                f"[Memory Debug] 用户 '{db_key}' 的当前印象:\n"
                f"------------------ 用户记忆 开始 ------------------\n"
                f"{mem_text if mem_text.strip() else '未找到该用户的记忆信息。'}\n"
                f"------------------- 用户记忆 结束 -------------------"

            )
            logger.info(log_message)

        # 2. 记录用户消息 (使用修正后的函数)
        processed_message = self._message_chain_to_text(event.message_obj.message)
        
        if processed_message:
            user_msg = f"[USER]: {processed_message}"
            async with self.lock:
                db.add_to_buffer(db_key, user_msg, max_len=self.HISTORY_WINDOW_SIZE)
                db.increment_counter(db_key)

    @filter.on_llm_response()
    async def collect_history_and_update(self, event: AstrMessageEvent, resp: LLMResponse):
        """响应后：记录AI消息 + 增加计数 + 检查是否触发更新"""
        db_key = self._get_db_key(event)
        if not db_key or not resp:
            return

        # [修正] 即使 completion_text 为空也要处理，因为AI可能通过工具等方式响应
        # 但我们只记录文本部分，所以依然依赖 completion_text
        ai_response_text = resp.completion_text if resp.completion_text else ""
        clean_resp = self._clean_text(ai_response_text)

        # 只有当清理后仍有文本内容时才记录，避免空消息
        if clean_resp:
            ai_msg = f"[ASSISTANT]: {clean_resp}"
            async with self.lock:
                db.add_to_buffer(db_key, ai_msg, max_len=self.HISTORY_WINDOW_SIZE)
                db.increment_counter(db_key)
                
                current_counter = db.get_counter(db_key)
                buffer_copy = list(db.get_buffer(db_key))

            if current_counter >= self.UPDATE_INTERVAL:
                if self.config.get("debug_memory"):
                    logger.info(f"Triggering memory update for {db_key}, counter: {current_counter}/{self.UPDATE_INTERVAL}")
                
                db.reset_counter(db_key)
                asyncio.create_task(self._update_memory_task(db_key, buffer_copy))

    def _extract_json(self, text: str):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
        return text

    async def _compress_text(self, text: str, provider) -> str:
        if not text or len(text) <= 600:
            return text

        logger.info(f"[Memory] Triggering compression for text (len: {len(text)})...")
        prompt = f"""请将以下关于用户的记忆片段进行“高保真压缩”。\n要求：\n1. 极度精简，保留最核心的事实、时间点和关键数据。\n2. 去除修饰词、废话和重复信息。\n3. 必须使用中文。\n4. 最终长度严格控制在 150 字以内。\n5. 直接输出结果，不要包含任何解释。\n\n需要压缩的内容：\n{text}"""

        try:
            response = await provider.text_chat(prompt=prompt, session_id=None, contexts=[])
            if response and response.completion_text:
                compressed = response.completion_text.strip()
                logger.info(f"[Memory] Compression successful: {len(text)} -> {len(compressed)}")
                return compressed
        except Exception as e:
            logger.error(f"[Memory] Compression failed: {e}")
        return text

    async def _update_memory_task(self, db_key: str, messages: List[str], reason: str = None):
        """后台任务：调用 LLM 更新记忆"""
        try:
            old_memory = db.get_memory(db_key)
            history_text = "\n".join(messages)
            
            reason_prompt = ""
            if reason:
                reason_prompt = f"\n\n【系统提示】本次更新由对话模型主动请求，并附带了以下注记：\n\"{reason}\"\n请自行判断将该信息归类到哪个字段其中之一最为合适\n"

            prompt = f"""你是一个专业的记忆归档员。你的任务是根据聊天记录更新用户的档案。\n**重要前提**：生成的这份档案将直接提供给【AI助手自己】（也就是未来的你）阅读，以便你更好地服务用户。\n\n当前关于该用户的记忆：\n{old_memory.to_text() if old_memory.to_text() else "尚无先前记忆。"}\n\n{reason_prompt}\n最近的聊天记录：\n{history_text}\n\n任务指令：\n结合【当前记忆】、【最近聊天记录】以及可能存在的【注记】，更新该用户的画像。\n1. **视角转换**：在描述用户与AI的互动、态度或共同经历时，**必须使用第二人称“你”来指代AI助手**。\n   - 错误示例：“用户不喜欢AI助手开玩笑。”\n   - 正确示例：“用户不喜欢**你**开玩笑。”\n   - 正确示例：“用户曾和**你**一起讨论过哲学问题。”\n2. **准确性**：保留旧记忆中仍然准确的信息，用新信息补充或修正。\n3. **精简**：如果某个字段没有新信息且旧记忆中不存在，返回 null。不要编造。\n4. **语言**：所有内容必须使用中文。\n\n你必须以严格的 JSON 格式输出结果，JSON 结构如下：\n{{\n    "disposition": "性格特征/人物画像",\n    "interests": "兴趣和爱好",\n    "doings": "ta目前正在做的事情",\n    "works": "职业或工作",\n    "wishes": "目标或心愿",\n    "worries": "烦恼或担忧",\n    "skills": "技能或专长",\n    "attitudes_to_model": "用户对【你】的态度 (请用'你'指代AI，例如：'觉得你很幽默')",\n    "experiences_with_model": "用户与【你】的共同经历 (请用'你'指代AI，例如：'曾向你请教知识')",\n    "extra_info": "任何其他能帮助【你】了解该用户的重要信息"\n}}\n\n仅输出 JSON 字符串，不要包含 Markdown 代码块（如 ```json ... ```）。"""

            if self.config.get("debug_memory"):
                logger.info(f"[Memory Debug] Update Request for {db_key}:\n{prompt}")

            provider_id = self.config.get("summary_provider_id", "")
            provider = self.context.get_provider_by_id(provider_id) if provider_id else self.context.get_using_provider()
            if not provider:
                logger.warn("No LLM provider available for memory update.")
                return

            response = await provider.text_chat(prompt=prompt, session_id=None, contexts=[])
            
            raw_result = response.completion_text
            if not raw_result:
                return
            
            if self.config.get("debug_memory"):
                logger.info(f"[Memory Debug] Update Response for {db_key}:\n{raw_result}")

            cleaned_result = self._extract_json(raw_result.strip())

            try:
                data = json.loads(cleaned_result)
                mem_result = MemoryResult(**data)
                
                new_memory = mem_result.get_updated_memory(old_memory)
                
                fields_to_check = ["experiences_with_you", "extra_info", "doings", "wishes"]
                for field in fields_to_check:
                    val = getattr(new_memory, field)
                    if val and len(val) > 600:
                        new_val = await self._compress_text(val, provider)
                        setattr(new_memory, field, new_val)

                db.update_memory(db_key, new_memory)
                logger.info(f"Memory updated for {db_key} successfully.")

            except json.JSONDecodeError:
                logger.error(f"JSON Parse failed for {db_key}, raw content: {raw_result}")
            except Exception as e:
                logger.error(f"Error processing memory update: {e}")

        except Exception as e:
            logger.error(f"Error in memory update task: {e}")