# config.py：存放敏感信息，被.gitignore忽略，不提交到GitHub
QWEN_TOKEN = "ms-bb133002-a51a-4276-9e1b-c95c7564a45a"  # 替换为自己的Token
# 部署相关配置（新增，便于后续部署修改）
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"  # 千问模型ID
SERVICE_PORT = 8000  # 后端服务端口（部署时可修改，默认8000）
MAX_HISTORY_ROUNDS = 3  # 多轮对话最大轮次（与第6天的截断逻辑一致）