# LLMJudge Dashboard

Vite + React + TypeScript SPA（设计 §10）。

```bash
# 需先启动 llmj-stack :8080
npm install --legacy-peer-deps
npm run dev     # :5173，代理到后端
npm run build   # 产出 dist/
```

Compose：`docker compose --profile monolith up` 后访问 `http://127.0.0.1:3000`。
