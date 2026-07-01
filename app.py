"""
重构后的 xiaozhi-server 启动入口 (FastAPI)
- 分离 ASR / TTS / QA 业务逻辑
- 移除所有 ESP32 相关依赖（WebSocket、OTA 等）
- 提供 /ask 接口：支持文本或语音输入，返回文本或合成语音
"""
import os
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
    def __init__(self, api_key: str, model: str = "deepseek-chat", system_prompt: str = "你是一个有用的助手"):
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.url = "https://api.deepseek.com/v1/chat/completions"

    async def ask(self, question: str, context: Optional[str] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "user", "content": context})
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
# ---------- 配置加载 ----------
def load_config() -> dict:
    """从环境变量加载 DeepSeek / 豆包 / 腾讯云配置"""
    from dotenv import load_dotenv
    load_dotenv()
    return {
        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "system_prompt": os.getenv("SYSTEM_PROMPT", "你是一个有用的助手"),
        },
        "doubao": {
            "app_id": os.getenv("DOUBAO_APP_ID", ""),
            "access_token": os.getenv("DOUBAO_ACCESS_TOKEN", ""),
            "cluster": os.getenv("DOUBAO_CLUSTER", "volcengine_streaming"),
        },
        "tencent": {
            "secret_id": os.getenv("TENCENT_SECRET_ID", ""),
            "secret_key": os.getenv("TENCENT_SECRET_KEY", ""),
            "voice_type": int(os.getenv("TENCENT_VOICE_TYPE", "1001")),  # 智瑜
            "codec": os.getenv("TENCENT_CODEC", "mp3"),
        },
        "server": {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8003")),
        },
    }

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

@app.post("/ask")
async def ask_endpoint(
    text: Optional[str] = Form(None, description="文本输入"),
    audio: Optional[UploadFile] = File(None, description="语音输入（WAV/MP3）"),
    response_format: Optional[str] = Form("text", description="输出格式：text 或 audio"),
    voice: Optional[str] = Form("alloy", description="TTS 声音（仅 audio 格式有效）"),
):
    """
    统一问答接口：
    - 若提供 audio 文件，进行语音识别 -> QA -> 按需合成语音
    - 若仅提供 text，直接 QA
    - response_format=audio 时返回合成语音（mp3），否则返回 JSON 文本
    """
    # 至少需要一种输入
    if not text and not audio:
        raise HTTPException(status_code=400, detail="请提供 text 或 audio 输入")

    # 1. 处理输入，获取提问文本
    question = ""
    if audio:
        # 读取上传的音频文件
        audio_bytes = await audio.read()
        # 可选: 用 ffmpeg 统一转成 16k mono wav，这里假设 asr 引擎能处理原始格式
        question = await asr_engine.transcribe(audio_bytes)
    if text:
        # 如果同时提供了两者，以 audio 识别结果为准，也可拼接，这里直接覆盖
        question = text if not question else question

    if not question.strip():
        raise HTTPException(status_code=400, detail="未能从输入中提取有效文本")

    # 2. 获取答案
    answer = await qa_engine.ask(question)

    # 3. 根据输出格式返回
    if response_format == "audio":
        audio_data = await tts_engine.synthesize(answer, voice=voice)
        return Response(content=audio_data, media_type="audio/mpeg")
    else:
        return JSONResponse(content={"answer": answer, "format": "text"})

# ---------- 启动入口 ----------
if __name__ == "__main__":
    config = load_config()
    host = config["server"]["host"]
    port = config["server"]["port"]
    print(f"启动 FastAPI 服务: http://{host}:{port}")
    print(f"接口文档: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)