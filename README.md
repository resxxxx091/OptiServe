# OptiServe 启动指南

## 前置要求

- Python ≥ 3.12，并安装 [uv](https://docs.astral.sh/uv/)
- Node.js ≥ 20.19（或 ≥ 22.12）
- Docker（用于跑 Redis / Milvus 中间件）

## 1. 配置环境变量

`.env` 放在仓库根目录，前后端共用。首次使用先复制模板：

```bash
cp .env.example .env
```

然后填好必填项：

- `OPTISERVE_API_TOKEN`：访问令牌，生成一条：
  `python -c "import secrets;print(secrets.token_urlsafe(32))"`
- `DEEPSEEK_API_KEY`：LLM 密钥
- `EMBEDDING_API_KEY`、`RERANK_API_KEY`：embedding 和 reranker 服务密钥
- `REDIS_PASSWORD`：自定义密码，注意 `REDIS_URL` 里的密码要和它一致

## 2. 启动中间件（Redis + Milvus）

```bash
cd deploy
docker compose --env-file ../.env up -d
```

Milvus 首次启动较慢（要拉 etcd、MinIO 等），等它健康了再起后端：

```bash
docker compose ps   # milvus 状态为 healthy 即可
```

## 3. 启动后端

```bash
cd backend
uv sync                     # 安装依赖
uv run python api/main.py   # 启动，默认 127.0.0.1:8000
```

开发模式下（`APP_ENV=development`）代码改动会自动重载。

注意：后端有启动闸门 —— LLM、embedding、reranker、Redis、Milvus 五个依赖
任何一个不通都会直接报错退出，不会降级启动。报错了先检查第 1、2 步。

## 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

打开 http://localhost:5173 ，在设置弹窗里填入后端地址和第 1 步生成的
`OPTISERVE_API_TOKEN` 即可使用。

后端接口文档：http://localhost:5173/docs （经 Vite 代理到后端 Swagger）。
