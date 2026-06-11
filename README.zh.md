# LLM 长期记忆（LLM Long-Term Memory）

为 LLM 提供长期记忆的层，其内部引擎是 **ENGRAM v1.1** 规范（[`ENGRAM_spec_v1_1.md`](ENGRAM_spec_v1_1.md)）的完整实现。运行时仅依赖文本嵌入模型（参考实现：EmbeddingGemma）与单文件数据库（SQLite）。生成式 LLM 仅在 **写入、读取后回复、做梦（整合）** 三处使用。

> 生成只在言语化的瞬间。判断全是距离。遗忘全是算术。破坏只发生在梦中。

- **文本是正本，向量是索引。** 记忆是简短的自洽命题（≤170 字）。`vec` 是可重新生成的派生缓存——即使嵌入模型消失，记忆也不会死。
- **三层**（MRL 截断 = 遗忘的分辨率）：**L1 情景**（768d f32，τ=7天）/ **L2 语义**（256d int8，τ=90天）/ **L3 图式**（128d int8，τ=3年）。
- **激活** `A = mass·2^(−Δt/τ)` 对余弦得分重新加权；同一性仅靠余弦距离阈值。整库 **<10MB**，检索为全量暴力余弦（`<1ms`，无向量数据库/FAISS 依赖）。
- 数据库**自描述**：`spec` 表内嵌完整规范全文。

产品名、文件结构、数据库名保持不变；仅算法、模式、参数与 UI 内容替换为 ENGRAM。

---

## 三层与激活/得分

| 层 | 容量 | 向量（MRL） | 半衰期 τ |
|----|------|------------|---------|
| **L1 情景** | 1000 | 768d f32 | 7 天 |
| **L2 语义** | 3000 | 256d int8 | 90 天 |
| **L3 图式** | 6000 | 128d int8 | 3 年 |

正文在所有层均无损；降级时只有检索键（向量）变粗。会话期间仅追加（非破坏）。

```text
A(now)   = mass × 2^( −max(0, now − last_access) / τ_tier )
回忆更新 : mass ← A(now);  若 now − last_bonus_at ≥ 3600: mass ← min(mass+1, 64)
A_abs    = ln(1 + A) / ln(1 + 64)
score(m) = max(0, cos(query, m)) × (α + (1−α) × A_abs),  α = 0.35
注入     : MMR（λ=0.3）选 5 条，连同表头 ≤1024 字
```

## 同一性阈值（文档—文档）

| cos | 判定 | 动作 |
|-----|------|------|
| ≥ 0.97 | 同一命题更新 | 旧记忆立墓碑，插入新记忆（再固化） |
| 0.85–0.97 | 冲突 | 两者保留 + `conflict` 队列（梦中裁决） |
| < 0.85 | 新建 | 插入，mass=1 |

文本完全一致时不新建行，视为回忆（mass+1）。机器迁移（无需 LLM）：晋升 A≥16，降级 A<4，淘汰按 A 升序物理删除 L3 溢出。**不存在不朽记忆**：因 mass≤64，沉默 6τ 后 A<1。

## DREAM（离线，破坏性操作仅在此）

聚类 → LLM 裁决（合并/拆分/不变）→ 物理删除被合并的源。1 次裁决 = 1 个事务。内容寻址指纹 + `dream_log` 防止空转。保留 8 代快照，提示词契约抑制虚构。

## 评测基准

```bash
python eval/run_eval.py   # 模拟嵌入/LLM + 虚拟时钟，无需 API key
```
确定性验证：激活衰减 / 回忆奖励 + 不应期 / 同一性阈值 / 层降级·晋升·淘汰 / 不朽性不存在 / 梦的合并。

## 启动

```bash
mkdir -p secrets && cp .env.example secrets/.env   # 设置 DEEPSEEK_API_KEY / HF_TOKEN
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
# Windows: start.bat  /  macOS·Linux: start.sh  → http://localhost:8501
python cli.py --seed --dream 3 --inspect
```

对话回合中，模型在回复的同时通过 `save_memory(text)` / `delete_memory(id)` 工具决定保存/删除。默认时区 `Asia/Tokyo`（可用 `MEMORY_TZ` 覆盖）。

详情见 [README.md](README.md) / [docs.html](docs.html) / [ENGRAM_spec_v1_1.md](ENGRAM_spec_v1_1.md)。

## 许可证

[MIT](LICENSE)
