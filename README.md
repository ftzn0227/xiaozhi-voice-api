# 小智FastAPI 部署指南
1. 系统结构图
<img width="563" height="569" alt="image" src="https://github.com/user-attachments/assets/4e617485-7ab0-4423-be8d-a90613e03c96" />
● 客户端：通过 HTTP/HTTPS 发送 POST 请求到 /ask，可携带文本或音频文件。
● FastAPI 应用：核心逻辑，集成三个外部 AI 服务。
● 外部服务：DeepSeek（LLM）、豆包（ASR）、腾讯云（TTS），均通过 API 调用。

2. 环境要求
● Python 3.9+
● 系统依赖：ffmpeg（用于音频格式转换）
● 网络：需能够访问 DeepSeek、火山引擎、腾讯云的公网 API
安装 ffmpeg：
pip install ffmpeg

3. 部署步骤
3.1 获取代码
git clone git@github.com:ftzn0227/xiaozhi-voice-api.git
3.2 创建虚拟环境
conda create -n xiaozhi-esp32-server python=3.10 -y
conda activate xiaozhi-esp32-server

# 添加清华源通道
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge

conda install libopus -y
conda install ffmpeg -y
3.3 安装依赖
pip install -r requirements.txt
pip install fastapi
pip install uvicorn
pip install python-multipart
pip install httpx
pip install tencentcloud-sdk-python
pip install python-dotenv
3.4 配置环境变量
创建 .env 文件：
# DeepSeek
DEEPSEEK_API_KEY=sk-xxxxxx
DEEPSEEK_MODEL=deepseek-chat
SYSTEM_PROMPT=你是一个全能的AI助手

# 豆包 ASR
DOUBAO_APP_ID=xxx
DOUBAO_ACCESS_TOKEN=xxx
DOUBAO_CLUSTER=volcengine_streaming

# 腾讯云 TTS
TENCENT_APP_ID=xxx
TENCENT_SECRET_ID=xxxx
TENCENT_SECRET_KEY=xxxx
TENCENT_VOICE_TYPE=1001
TENCENT_CODEC=mp3

# 服务端口
HOST=0.0.0.0
PORT=8003
3.5 启动服务（开发/测试）
1. 启动
python app.py

2. 测试
curl -X POST http://localhost:8003/ask -F "text=用一句话介绍你自己"
