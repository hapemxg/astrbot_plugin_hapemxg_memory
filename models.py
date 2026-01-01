from enum import StrEnum
from pydantic import BaseModel, Field

class MemoryAboutUser(BaseModel):
    disposition: str | None = Field(None, description="性格")
    interests: str | None = Field(None, description="兴趣爱好")
    doings: str | None = Field(None, description="正在做的事情")
    works: str | None = Field(None, description="工作/职业")
    wishes: str | None = Field(None, description="愿望/目标")
    worries: str | None = Field(None, description="担忧/烦恼")
    skills: str | None = Field(None, description="技能/专长")
    attitudes_to_you: str | None = Field(None, description="对你的态度")
    experiences_with_you: str | None = Field(None, description="与你的经历")
    extra_info: str | None = Field(None, description="其他补充信息, 若无可不填")

    def to_text(self) -> str:
        parts = []
        if self.disposition:
            parts.append(f"性格: {self.disposition}")
        if self.interests:
            parts.append(f"兴趣爱好: {self.interests}")
        if self.doings:
            parts.append(f"正在做的事情: {self.doings}")
        if self.works:
            parts.append(f"工作/职业: {self.works}")
        if self.wishes:
            parts.append(f"愿望/目标: {self.wishes}")
        if self.worries:
            parts.append(f"担忧/烦恼: {self.worries}")
        if self.skills:
            parts.append(f"技能/专长: {self.skills}")
        if self.attitudes_to_you:
            parts.append(f"对你的态度: {self.attitudes_to_you}")
        if self.experiences_with_you:
            parts.append(f"与你的经历: {self.experiences_with_you}")
        if self.extra_info:
            parts.append(f"其他补充信息: {self.extra_info}")
        return "\n".join(parts)


class MemoryResult(BaseModel):
    disposition: str | None = Field(None, description="性格")
    interests: str | None = Field(None, description="兴趣爱好")
    doings: str | None = Field(None, description="正在做的事情")
    works: str | None = Field(None, description="工作/职业")
    wishes: str | None = Field(None, description="愿望/目标")
    worries: str | None = Field(None, description="担忧/烦恼")
    skills: str | None = Field(None, description="技能/专长")
    attitudes_to_model: str | None = Field(
        None, description="对聊天中的AI助手的态度, 若消息记录中没有AI助手的则保持不变"
    )
    experiences_with_model: str | None = Field(
        None, description="与聊天中的AI助手的经历, 若消息记录中没有AI助手的则保持不变"
    )
    extra_info: str | None = Field(None, description="其他补充信息, 若无可不填")

    def get_memory(self) -> MemoryAboutUser:
        return MemoryAboutUser(
            disposition=self.disposition,
            interests=self.interests,
            doings=self.doings,
            works=self.works,
            wishes=self.wishes,
            worries=self.worries,
            skills=self.skills,
            attitudes_to_you=self.attitudes_to_model,
            experiences_with_you=self.experiences_with_model,
            extra_info=self.extra_info,
        )

    def get_updated_memory(self, old_memory: MemoryAboutUser) -> MemoryAboutUser:
        # 辅助函数：如果新值存在则使用新值，否则保留旧值
        def merge(new_val, old_val):
            # 策略：如果 LLM 返回 None，则认为它没有提取到新信息或不想修改，保留旧值
            return new_val if new_val is not None else old_val

        return MemoryAboutUser(
            disposition=merge(self.disposition, old_memory.disposition),
            interests=merge(self.interests, old_memory.interests),
            doings=merge(self.doings, old_memory.doings),
            works=merge(self.works, old_memory.works),
            wishes=merge(self.wishes, old_memory.wishes),
            worries=merge(self.worries, old_memory.worries),
            skills=merge(self.skills, old_memory.skills),
            attitudes_to_you=merge(self.attitudes_to_model, old_memory.attitudes_to_you),
            experiences_with_you=merge(self.experiences_with_model, old_memory.experiences_with_you),
            extra_info=merge(self.extra_info, old_memory.extra_info),
        )