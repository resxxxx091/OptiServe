# OptiServe Frontend

OptiServe 智能客服系统的调试工作台，单页 Vue 3 应用，连接 `backend/api/main.py` 起的 FastAPI 服务。

## 视图

| 视图 | 用途 | 后端接口 |
|---|---|---|
| 对话 | 发一条真实请求，看意图、路由、耗时 | `POST /chat` |
| 知识库 | 检索链路验证、文档导入、文件上传、Skills 查看与热加载 | `POST /search`、`POST /knowledge/add`、`POST /knowledge/upload`、`GET /knowledge/stats`、`GET /skills`、`POST /skills/reload` |

顶栏常驻健康指示，来自 `GET /health`；运行状态卡在 `GET /monitor`。

## 后端地址

浏览器侧基准路径是 `/api/python`，由 `VITE_PYTHON_API_URL` 覆盖：

```bash
VITE_PYTHON_API_URL=http://localhost:8000 npm run dev
```

不设这个变量时走开发代理：Vite 把 `/api/python` 反代到 `http://localhost:8000`（`vite.config.js`）。这是目前唯一的接法——仓库里已没有 nginx 反代配置，真要上线托管得自备一个把 `/api/python` 指到后端的反代。

注意 `import.meta.env` 是**构建期**内联的，运行期改反代改不动它。要保持一份产物适配任意部署位置，就沿用默认的相对路径 `/api/python`，把上游交给托管方的反代决定。

## 本地运行

```bash
npm install
npm run dev      # http://localhost:5173
```

后端需要先起得来：`api/main.py` 在启动阶段对 LLM / embedding / reranker / Redis / Milvus 逐个真实探测，任一不通直接抛错终止启动，因此前端拿到 `503` 通常意味着后端没通过启动闸门，不是前端问题。

## Docker 部署

只有中间件进容器：仓库根 `deploy/docker-compose.yml` 编排 Redis + Milvus（etcd / MinIO 是 Milvus standalone 的内部依赖），**不含前后端**——后端本机 `uvicorn api/main.py`，前端就是上面那节 `npm run dev`。

```bash
cd deploy                                   # 工作目录 = deploy/
docker compose --env-file ../backend/.env up -d
docker compose down
```

`--env-file` 不能省：`${REDIS_PASSWORD}` 的插值只认它。栈里没有需要构建的服务，别加 `--build`。

前端镜像的构建配方（`Dockerfile` + `nginx.conf`）曾在 `deploy/` 下，2026-09-21 删除：那是六服务旧栈的遗留，当时已没有任何 `build:` 引用它，且 nginx 反代的服务名 `optiserve` 在栈里不存在。将来要容器化重新写一份即可，注意构建上下文必须覆盖到 nginx 配置所在目录。

## 浏览器存的设置

`localStorage` 键 `optiserve.frontend.settings`，只放用户 ID 与会话 ID 两项。清空会话或改这两个输入框会写回。
