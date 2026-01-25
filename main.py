# main.py (完整修改版)

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

    def _resolve_db_key(self, event: AstrMessageEvent, target_user_id: Optional[str]) -> str:
        final_user_id = target_user_id if target_user_id else event.get_sender_id()
        if self.config.get("memory_scope", False):
            group_id = event.message_obj.group_id
            if group_id:
                return f"{group_id}_{final_user_id}"
        return final_user_id

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        for pattern in self.clean_patterns:
            try:
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
        
        if user_id and user_id != sender_id:
            global_config = getattr(self.context, "_config", {})
            if not global_config:
                 global_config = getattr(self.context, "config", {})
            
            admins = global_config.get("admins_id", [])
            
            if sender_id not in admins:
                yield event.plain_result("只有管理员可以查看其他用户的记忆状态。")
                return

        db_key = self._resolve_db_key(event, user_id)
        
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
    @memory_group.command("compress")
    async def memory_compress_cmd(self, event: AstrMessageEvent, user_id: str = None):
        """[管理员] 手动压缩指定用户的记忆，去除冗余"""
        db_key = self._resolve_db_key(event, user_id)
        memory = db.get_memory(db_key)
        
        provider_id = self.config.get("summary_provider_id", "")
        provider = self.context.get_provider_by_id(provider_id) if provider_id else self.context.get_using_provider()
        
        if not provider:
            yield event.plain_result("错误：未找到可用的 LLM Provider，无法执行压缩。")
            return

        yield event.plain_result(f"正在对 {user_id if user_id else '你'} 的记忆进行深度压缩整理，请稍候...")

        new_memory, logs = await self._batch_compress_memory(memory, provider, force=True)
        
        db.update_memory(db_key, new_memory)
        
        log_str = "\n".join(logs)
        yield event.plain_result(f"压缩完成！\n{log_str}")

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
        """
        当用户在对话中透露了重要的个人信息（如：姓名、年龄、喜好、厌恶、职业、人际关系、重要经历等）时，调用此工具以立即更新用户的长期记忆档案。
        注意：请勿在普通闲聊或无关紧要的对话中滥用此工具。

        Args:
            reason (string): 简要描述需要被记忆的具体信息摘要。例如：“用户提到他养了一只叫Luna的猫”、“用户表示不喜欢吃香菜”。
        """
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

    @filter.on_llm_request(priority=-2)
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        """请求前：注入记忆 + 记录用户消息 + 增加计数"""
        db_key = self._get_db_key(event)
        if not db_key:
            return

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

        ai_response_text = resp.completion_text if resp.completion_text else ""
        clean_resp = self._clean_text(ai_response_text)

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

    async def _compress_text(self, text: str, provider, max_length: int = 50) -> str:
        """
        [底层原子方法] 只压缩传入的这一段文本，不涉及任何其他上下文。
        """
        if not text or len(text) < max_length * 0.8: 
            return text

        prompt = f"""任务：文本压缩
要求：
1. 将下方【待压缩文本】的内容精简到 {max_length} 字以内。
2. 只保留核心事实、关键时间点和重要名词。
3. 去除无意义的修饰词、口语词和重复内容。
4. 直接输出压缩后的结果，不要包含"好的"、"压缩结果："等废话。

【待压缩文本】：
{text}"""

        try:
            response = await provider.text_chat(prompt=prompt, session_id=None, contexts=[])
            if response and response.completion_text:
                compressed = response.completion_text.strip()
                compressed = compressed.strip('"').strip("'")
                
                if len(compressed) >= len(text) or not compressed:
                    return text
                    
                logger.info(f"[Memory] Compressed: {len(text)} -> {len(compressed)} chars.")
                return compressed
        except Exception as e:
            logger.error(f"[Memory] Compression API failed: {e}")
        
        return text

    async def _batch_compress_memory(self, memory: MemoryAboutUser, provider, force: bool = False) -> (MemoryAboutUser, List[str]):
        """
        [逻辑编排方法] 批量并发压缩内存对象中的字段
        :param force: 是否强制压缩（手动模式下为 True，忽略长度阈值，只要超标就压）
        :return: (更新后的内存对象, 变更日志列表)
        """
        rules = {
            # --- 核心大段文本 ---
            "experiences_with_you": {"trigger": 150, "target": 80}, # 共同经历：最容易变长，允许稍长
            "extra_info":           {"trigger": 100, "target": 60}, # 补充信息：容易堆积杂项
            
            # --- 状态/描述类 ---
            "attitudes_to_you":     {"trigger": 100,  "target": 60}, # 态度
            "disposition":          {"trigger": 100,  "target": 60}, # 性格
            
            # --- 短语/列表类 (之前缺失的补回来了) ---
            # 这里的阈值设置稍高，为了保护"列表"格式不被过度概括
            "interests":            {"trigger": 160, "target": 90}, # 兴趣：防止写成小作文
            "skills":               {"trigger": 160, "target": 90}, # 技能：同上
            
            # --- 短文本类 ---
            "doings":               {"trigger": 60,  "target": 40}, # 正在做的事：时效性强，需精简
            "wishes":               {"trigger": 60,  "target": 40}, # 愿望
            "worries":              {"trigger": 60,  "target": 40}, # 烦恼
            "works":                {"trigger": 60,  "target": 20}, # 工作：通常就是职业名，很短
        }

        tasks = []
        logs = []

        async def _task_wrapper(field, text, target):
            new_text = await self._compress_text(text, provider, max_length=target)
            if new_text != text:
                return field, new_text
            return None

        for field, rule in rules.items():
            val = getattr(memory, field)
            if not val:
                continue

            threshold = 0 if force else rule["trigger"]
            target_len = rule["target"]

            if (force and len(val) > target_len) or (not force and len(val) > threshold):
                logger.info(f"[Memory] Plan to compress '{field}': len {len(val)} -> target {target_len}")
                tasks.append(_task_wrapper(field, val, target_len))

        if not tasks:
            return memory, ["没有字段需要压缩。"]

        results = await asyncio.gather(*tasks)

        changes_count = 0
        for res in results:
            if res:
                field_name, new_val = res
                old_len = len(getattr(memory, field_name))
                new_len = len(new_val)
                setattr(memory, field_name, new_val)
                logs.append(f"[{field_name}] {old_len}字 -> {new_len}字")
                changes_count += 1
        
        if changes_count == 0:
            logs.append("虽然尝试了压缩，但LLM认为当前内容已是最简或压缩失败。")

        return memory, logs

    async def _update_memory_task(self, db_key: str, messages: List[str], reason: str = None):
        """后台任务：调用 LLM 更新记忆"""
        try:
            old_memory = db.get_memory(db_key)
            history_text = "\n".join(messages)
            
            reason_prompt = ""
            if reason:
                reason_prompt = f"\n\n【系统提示】本次更新由对话模型主动请求，并附带了以下注记：\n\"{reason}\"\n请自行判断将该信息归类到哪个字段其中之一最为合适\n"

            prompt = f"""你是一个专业的记忆归档员。你的任务是根据聊天记录更新这一名用户的档案。\n**重要前提**：生成的这份档案将直接提供给【AI助手自己】（也就是未来的你）阅读，以便你更好地服务用户。\n\n当前关于这名用户的记忆：\n{old_memory.to_text() if old_memory.to_text() else "尚无先前记忆。"}\n\n{reason_prompt}\n最近的聊天记录（请忽略掉和报错有关的内容）：\n{history_text}\n\n任务指令：\n结合【当前记忆】、【最近聊天记录】以及可能存在的【注记】，更新这名用户的画像。\n1. **视角转换**：在描述用户与AI的互动、态度或共同经历时，**必须使用第二人称“你”来指代AI助手**。\n   - 错误示例：“这个用户不喜欢AI助手开玩笑。”\n   - 正确示例：“这个用户不喜欢**你**开玩笑。”\n   - 正确示例：“这个用户曾和**你**一起讨论过哲学问题。”\n2. **准确性**：保留旧记忆中仍然准确的信息，用新信息补充或修正。\n3. **精简**：如果某个字段没有新信息且旧记忆中不存在，返回 null。不要编造。\n4. **语言**：所有内容必须使用中文。\n\n你必须以严格的 JSON 格式输出结果，JSON 结构如下：\n{{\n    "disposition": "性格特征/人物画像",\n    "interests": "兴趣和爱好",\n    "doings": "ta目前正在做的事情",\n    "works": "职业或工作",\n    "wishes": "目标或心愿",\n    "worries": "烦恼或担忧",\n    "skills": "技能或专长",\n    "attitudes_to_model": "这个用户对【你】的态度 (请用'你'指代AI，例如：'觉得你很幽默')",\n    "experiences_with_model": "这个用户与【你】的共同经历 (请用'你'指代AI，例如：'曾向你请教知识')",\n    "extra_info": "任何其他能帮助【你】了解这个用户的重要信息"\n}}\n\n仅输出 JSON 字符串，不要包含 Markdown 代码块（如 ```json ... ```）。"""

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
                
                # 1. 先合并新旧记忆
                intermediate_memory = mem_result.get_updated_memory(old_memory)
                
                # 2. 调用新的批量压缩流程 (自动模式 force=False)
                final_memory, _ = await self._batch_compress_memory(intermediate_memory, provider, force=False)

                # 3. 保存最终结果
                db.update_memory(db_key, final_memory)
                logger.info(f"Memory updated for {db_key} successfully.")

            except json.JSONDecodeError:
                logger.error(f"JSON Parse failed for {db_key}, raw content: {raw_result}")
            except Exception as e:
                logger.error(f"Error processing memory update: {e}")

        except Exception as e:
            logger.error(f"Error in memory update task: {e}")