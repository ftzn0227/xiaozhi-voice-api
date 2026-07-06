"""
意图路由模块
- classify_intent: 将用户输入分类为 simple_qa / multi_step
- multi_step_qa: 对复杂问题生成计划+推理+答案
- get_suggestions: 根据对话生成后续问题推荐
"""
import json
import httpx
import logging
import traceback
from config import load_config
logger = logging.getLogger(__name__)
# ---------- 提示词 ----------
INTENT_CLASSIFY_PROMPT = """
你是一个意图分类器。分析用户输入，仅返回一个 JSON：
{{"intent": "simple_qa" 或 "multi_step", "reason": "简短分类理由"}}

- simple_qa: 简单问题，不需要调用外部工具或多步推理，可直接回答。
- multi_step: 需要多步推理、计算或调用外部工具（如天气、搜索）才能回答。

用户输入: {user_message}
"""

MULTI_STEP_PROMPT = """
你是问题解决专家。请按以下 JSON 格式回答：
{{
  "plan": ["步骤1", "步骤2", ...],
  "reasoning": ["步骤1的思考", "步骤2的思考"],
  "answer": "最终答案"
}}
用户问题: {question}
"""

SUGGESTION_PROMPT = """
根据以下对话，生成3个用户可能接着问的问题（每行一个，不要编号）：
用户: {last_user}
助手: {last_answer}
"""

# ---------- 工具函数 ----------
async def call_deepseek(messages: list, temperature: float = 0.7, json_mode: bool = False) -> str:
    """调用 DeepSeek API，返回模型输出的文本内容"""
    config = load_config()
    api_key = config["deepseek"]["api_key"]
    model = config["deepseek"].get("model", "deepseek-chat")
    url = "https://api.deepseek.com/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

# ---------- 核心函数 ----------
async def classify_intent(user_message: str) -> dict:
    """意图分类"""
    config = load_config()
    if "deepseek" not in config or not config["deepseek"].get("api_key"):
        logger.error("DeepSeek API key 未配置，意图分类不可用")
        raise ValueError("DeepSeek 配置缺失")
    prompt = INTENT_CLASSIFY_PROMPT.format(user_message=user_message)
    try:
        content = await call_deepseek(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            json_mode=True
        )
        return json.loads(content)
    except Exception as e:
        # 降级策略
        print(f"意图分类失败，降级为 simple_qa: {e}")
        return {"intent": "simple_qa", "reason": "fallback"}

async def multi_step_qa(question: str, history: list = None) -> dict:
    """处理多步问题，返回包含 plan/reasoning/answer 的字典"""
    messages = []
    if history:
        # 添加简要历史（避免过长，可以截取最后几轮）
        messages.append({"role": "system", "content": "以下为历史对话："})
        messages.extend(history[-6:])   # 只取最近3轮 (用户+助手各3条)
    messages.append({"role": "user", "content": MULTI_STEP_PROMPT.format(question=question)})

    content = await call_deepseek(messages, temperature=0.3, json_mode=True)
    return json.loads(content)

async def get_suggestions(history: list, last_answer: str) -> list:
    """生成3个后续推荐问题"""
    if not history:
        return []
    last_user = history[-1]["content"] if history[-1]["role"] == "user" else ""
    prompt = SUGGESTION_PROMPT.format(last_user=last_user, last_answer=last_answer)
    try:
        content = await call_deepseek(
            [{"role": "user", "content": prompt}],
            temperature=0.8
        )
        suggestions = [line.strip("-•1234567890. ").strip() for line in content.split("\n") if line.strip()]
        return suggestions[:3]
    except Exception as e:
        print(f"生成推荐问题失败: {e}")
        print(f"生成推荐问题失败详情: {traceback.format_exc()}")
        return []