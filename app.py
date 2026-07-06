"""
重构后的 xiaozhi-server 启动入口 (FastAPI)
- 分离 ASR / TTS / QA 业务逻辑
- 移除所有 ESP32 相关依赖（WebSocket、OTA 等）
- 提供 /ask 接口：支持文本或语音输入，返回文本或合成语音
"""
import os
import traceback
import uuid
import tempfile
from io import BytesIO
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import httpx
from contextlib import asynccontextmanager

from typing import Dict, List

from config import load_config
from intent_router import classify_intent, multi_step_qa, get_suggestions
from conversation import ConversationMemory
# ---------- 抽象业务接口 ----------
class BaseASR:
    async def transcribe(self, audio_data: bytes) -> str:
        raise NotImplementedError

class BaseQA:
    async def ask(self, question: str, context: Optional[str] = None) -> str:
        raise NotImplementedError

class BaseTTS:
    async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
        raise NotImplementedError

# ---------- 基于 OpenAI 的实现示例 ----------
class DoubaoASR(BaseASR):
    def __init__(self, app_id: str, access_token: str, cluster: str = "volcengine_streaming"):
        self.app_id = app_id
        self.access_token = access_token
        self.cluster = cluster
        self.url = "https://openspeech.bytedance.com/api/v1/asr"

    async def transcribe(self, audio_data: bytes) -> str:
        # 豆包 ASR 需要先上传音频文件，这里使用一句话识别接口（仅示例）
        headers = {
            "Authorization": f"Bearer; {self.access_token}",
        }
        files = {
            "audio": ("audio.wav", audio_data, "audio/wav"),
        }
        params = {
            "appid": self.app_id,
            "cluster": self.cluster,
            "format": "wav",
            "rate": "16000",
            "bits": "16",
            "channel": "1",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.url, params=params, headers=headers, files=files, timeout=30.0)
            resp.raise_for_status()
            result = resp.json()
            # 解析识别结果（根据实际返回结构调整）
            if result.get("code") == 1000:
                return result["result"][0]["text"]
            else:
                raise Exception(f"豆包 ASR 错误: {result.get('message')}")

class DeepSeekQA(BaseQA):
    # 添加历史对话
    def __init__(self, api_key: str, model: str = "deepseek-chat", system_prompt: str = "你是一个有用的助手"):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.url = "https://api.deepseek.com/v1/chat/completions"

    async def ask(self, question: str, history: List[dict] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(self.url, json=payload, headers=headers, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

class TencentTTS(BaseTTS):
    def __init__(self, secret_id: str, secret_key: str, voice_type: int = 1001, codec: str = "mp3"):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.voice_type = voice_type
        self.codec = codec
        self.url = "https://tts.tencentcloudapi.com/"

    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        # 生成签名等参数（简化为使用腾讯云 SDK，这里用 httpx 示例）
        from tencentcloud.common import credential
        from tencentcloud.tts.v20190823 import tts_client, models

        cred = credential.Credential(self.secret_id, self.secret_key)
        client = tts_client.TtsClient(cred, "ap-guangzhou")  # 地域可按需配置

        req = models.TextToVoiceRequest()
        req.Text = text
        req.VoiceType = self.voice_type
        req.Codec = self.codec
        req.SessionId = str(uuid.uuid4())

        # 异步调用腾讯云 SDK 需使用异步客户端，这里简化为同步调用（可用 run_in_executor）
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(None, client.TextToVoice, req)

        # 返回音频数据（Base64 解码）
        audio_data = base64.b64decode(resp.Audio)
        return audio_data
# ---------- 工具函数 ----------
def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用（处理音频转换需要，非强制）"""
    import shutil
    return shutil.which("ffmpeg") is not None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化引擎
    config = load_config()
    init_engines(config)
    if not check_ffmpeg():
        print("警告: 未找到 ffmpeg，某些音频格式转换可能失败")
    yield

# ---------- 创建 FastAPI 应用 ----------
app = FastAPI(
    title="Xiaozhi AI Server (Refactored)",
    version="2.0",
    lifespan=lifespan   # 关键：绑定 lifespan
)

# 初始化业务组件（延迟加载，因为需要配置）
asr_engine: Optional[BaseASR] = None
qa_engine: Optional[BaseQA] = None
tts_engine: Optional[BaseTTS] = None

def init_engines(config: dict):
    global asr_engine, qa_engine, tts_engine

    # DeepSeek QA
    qa_engine = DeepSeekQA(
        api_key=config["deepseek"]["api_key"],
        model=config["deepseek"]["model"],
        system_prompt=config["deepseek"]["system_prompt"],
    )

    # 豆包 ASR
    asr_engine = DoubaoASR(
        app_id=config["doubao"]["app_id"],
        access_token=config["doubao"]["access_token"],
        cluster=config["doubao"]["cluster"],
    )

    # 腾讯云 TTS
    tts_engine = TencentTTS(
        secret_id=config["tencent"]["secret_id"],
        secret_key=config["tencent"]["secret_key"],
        voice_type=config["tencent"]["voice_type"],
        codec=config["tencent"]["codec"],
    )

# ---------- /ask 接口 ----------
class AskResponse(BaseModel):
    answer: str
    format: str = "text"

memory = ConversationMemory(max_history=10, ttl=3600)
@app.post("/ask")   # 兼容老接口
@app.post("/chat")  # 新接口
async def chat(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    response_format: str = Form("text"),
    voice: str = Form("alloy"),
):
    # --- 1. 提取问题文本 ---
    if audio:
        audio_bytes = await audio.read()
        question = await asr_engine.transcribe(audio_bytes)
    elif text:
        question = text.strip()
    else:
        raise HTTPException(400, "请提供 text 或 audio")

    # --- 2. 会话处理 ---
    if not session_id:
        session_id = str(uuid.uuid4())

    history = memory.get_history(session_id)

    # --- 3. 意图分类 ---
    try:
        intent_info = await classify_intent(question)
    except Exception as e:
        print(f"意图分类失败: {e}")
        print(f"意图分类失败详情: {traceback.format_exc()}")
        intent_info = {"intent": "simple_qa", "reason": "error"}

    # --- 4. 按意图处理 ---
    plan = None
    if intent_info["intent"] == "multi_step":
        result = await multi_step_qa(question, history)
        answer = result["answer"]
        plan = result.get("plan")
    else:
        answer = await qa_engine.ask(question, history)

    # --- 5. 更新记忆 ---
    memory.add_message(session_id, "user", question)
    memory.add_message(session_id, "assistant", answer)

    # --- 6. 推荐问题 ---
    suggestions = await get_suggestions(history, answer) if config.get("enable_suggestions") else []

    # --- 7. 构建响应 ---
    if response_format == "audio":
        audio_content = await tts_engine.synthesize(answer, voice=voice)
        return Response(content=audio_content, media_type="audio/mpeg")
    else:
        resp_dict = {
            "session_id": session_id,
            "answer": answer,
            "intent": intent_info["intent"],
        }
        if plan:
            resp_dict["plan"] = plan
        if suggestions:
            resp_dict["suggestions"] = suggestions
        return JSONResponse(content=resp_dict)

# ---------- 启动入口 ----------
if __name__ == "__main__":
    config = load_config()
    host = config["server"]["host"]
    port = config["server"]["port"]
    print(f"启动 FastAPI 服务: http://{host}:{port}")
    print(f"接口文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)