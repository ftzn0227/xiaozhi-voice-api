import os

class Settings:
    """配置类"""
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
            "enable_suggestions": bool(os.getenv("ENABLE_SUGGESTIONS", "True"))
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
        "enable_suggestions": bool(os.getenv("ENABLE_SUGGESTIONS", "True"))
    }