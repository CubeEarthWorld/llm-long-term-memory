# LLM 长期记忆（LLM Long-Term Memory）

一个受人类记忆启发的轻量级 LLM 长期记忆层。  
**仅在写入时（信息抽取）** 和 **Dreaming（记忆整合）** 时调用 LLM。  
回忆、评分、聚类和归档管理完全在本地通过 Embedding 和数值计算完成——无需向量数据库或 FAISS。

- **双层 SQLite 存储**：**active（热数据，≤1000 条，768 维向量）** 和 **archive（冷数据，≤5000 条，仅保存 256 维粗向量）**。总容量有界（约数十 MB）。
- 搜索使用 **768 维完整向量**。**256 维粗向量** 由完整向量实时导出，用于聚类、多样性调节、联想扩散和归档匹配。
- 取回的 memory pack 最大 **1024 字符**。
- 每条记忆仿照人类记忆分离为三轴：**`w`**（重要度）、**`confidence`**（可信度）、**`S`**（稳定度）。
- 每条记忆附带 **IANA 时区** 时间戳，在回忆和 Dreaming 时传递给 LLM。

---

## 目录

- [数据结构](#数据结构)
- [保持、回忆与遗忘](#保持回忆与遗忘)
- [Dreaming（记忆整合）](#dreaming记忆整合)
- [评估基准](#评估基准)
- [快速开始](#快速开始)
- [CLI 用法](#cli-用法)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 数据结构

```sql
CREATE TABLE memories (              -- active（热数据）
  id TEXT PRIMARY KEY,               -- uuid7
  text TEXT, v_full BLOB,            -- 768 维（粗向量实时导出）
  w REAL, provenance TEXT, confidence REAL,  -- 重要度 / 'user'|'inferred' / 可信度
  freq REAL, stability REAL,         -- 累计访问次数 / 稳定度 S（秒）
  accessed_at_unix REAL,             -- 最后访问时间 → 衰减基准
  updated_at_unix REAL,              -- 内容版本时间戳 → 回忆/Dream 使用
  timezone TEXT, cluster_id TEXT,
  source_ids TEXT, summary_of TEXT, dream_action TEXT, created_by_dream_id TEXT  -- Dream 血缘
);

CREATE TABLE archive (               -- cold（ savings 用）
  id TEXT PRIMARY KEY,
  text TEXT, v_coarse BLOB,          -- 256 维粗向量（仅用于再出现匹配）
  w REAL, provenance TEXT, confidence REAL,
  last_r REAL, archived_at_unix REAL, text_hash TEXT, timezone TEXT
);

CREATE TABLE clusters (
  id TEXT PRIMARY KEY,
  last_dreaming_unix REAL            -- 创建 UNIX 时间；每次 Dream 时更新
);
```

质心 / 大小 / medoid 按需从聚类成员实时导出（不存储）。

---

## 保持、回忆与遗忘

稳定度 `S`（秒）与可提取率 `r`（FSRS 风格）分离。遗忘遵循**幂律**，每次成功回忆都会增长 `S`（在即将遗忘时回忆增长更多——间隔效应/测试效应、Jost 法则）。

```text
S0           = stab_base * (1 + kappa*w) * (labile_frac + (1-labile_frac)*confidence)
r(t)         = (1 + (now - accessed_at) / S) ^ (-forget_beta)        # 遗忘曲线（幂律）
强化(回忆):   S = S * (1 + stab_growth_c*(1-r)); freq += reinforce_inc; accessed_at = now
               （S 与 freq 通过 max_stability_seconds / max_freq 截断，防止无限增长）
score        = alpha*cos + beta*r + delta*w + eta*log1p(freq) + zeta*confidence  (+ 联想扩散)
回忆门控     : gate_w_cos*cos + gate_w_r*r (+noise) >= gate_theta    # 功能性遗忘
```

遗忘分两段且容量有界：

1. **active → archive**：当 `r` 低于 `r_archive_floor`（超过宽限期后）时，记忆被移入 archive——正常回忆无法访问，但作为 *savings* 保留。高 `w` + 高 `confidence` 的记忆受保护，直到 `r_hard_floor`。容量超限时，优先退避最低 `r` 的记忆。
2. **archive → 永久删除**：archive 超过 `archive_cap` 时，从最低 `last_r` 开始永久删除。
3. **savings（再出现）**：相同或相似文本再次出现时（粗向量 cosine ≥ `tau_savings` 或 text_hash 匹配），以稳定度先发优势（`× savings_gain`）恢复到 active——再学习节省。
4. **干扰**：被回忆的记忆会轻微衰减同聚类中未被复现竞争者的稳定度（提取诱发遗忘）。

---

## Dreaming（记忆整合）

睡眠类比式的整理处理。按需执行；LLM 处理高优先级聚类，选择 **合并(merge)**、**分割(split)** 或 **维持(none)**。

- **优先级**（计算得出）= `w_size*log(1+size) + w_spread*(spread/norm) + w_age*(since_dream/norm) + w_disp*dispersion`
  - `spread` = 聚类内 `updated_at_unix` 的时间跨度；`since_dream` = 自 `last_dreaming_unix` 以来的经过时间。
  - 小于 `dream_min_size` 或处于 `dream_min_interval` 冷却期内的聚类被跳过。
- LLM 接收成员内容、重要度、**带时区的时间戳** 和可提取率 `r`。输出记忆替换输入记忆。
  - **merge** = 将相关记忆减少并 gist 化（情景记忆 → 语义记忆）。
  - **split** = 将多个事实拆分为独立记忆。
  - 陈旧内容会根据当前时间更新（例如『7月计划旅行』→『2026年7月已旅行』）。冲突以较新时间戳为准。
- 整合后记忆的 **稳定度不超过最佳来源**（`S = max(member S)`）。血缘通过 `source_ids` / `created_by_dream_id` 保留。

---

## 评估基准

`eval/run_eval.py` 使用虚拟时钟 + Mock Embedding / Mock LLM 运行确定性保真度基准测试——**无需下载模型或 API 密钥**。

```bash
python eval/run_eval.py
```

验证 8 种人类记忆行为：

| 场景 | 验证内容 |
|---|---|
| 间隔效应 > 集中效应 | 间隔回忆比集中回忆更能增长稳定度 |
| 幂律遗忘尾部 | 可提取率随时间下降，但保留重尾 |
| 回忆门控 | 功能性遗忘阻止对衰减记忆的弱线索提取 |
| 稳定度增长 | 每次成功提取都会增加稳定度 |
| archive → savings 恢复 | 归档记忆恢复时带有稳定度先发优势 |
| archive 容量有界 | archive 遵守 `archive_cap`（总容量有界） |
| 三轴保护 | 高可信度、高重要度记忆抵抗驱逐 |
| 联想扩散 | 聚类兄弟节点获得联想扩散加成 |

---

## 快速开始

### 1. 配置 API 密钥

复制示例文件并填入真实密钥。`secrets/` 目录已被 git 忽略，密钥不会泄露到版本控制中。

```bash
mkdir -p secrets
cp .env.example secrets/.env
# 编辑 secrets/.env 填入密钥
```

- **DeepSeek**（默认）：设置 `DEEPSEEK_API_KEY`
- **Gemini**（可选）：设置 `LLM_PROVIDER=gemini` 和 `GEMINI_API_KEY`
- **Hugging Face**（下载 EmbeddingGemma）：设置 `HF_TOKEN`

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 启动

Windows: `start.bat` / macOS & Linux: `start.sh` → 浏览器打开 `http://localhost:8501`。  
可以在对话标签页点击 “💤 Dream” 按钮执行整理。

---

## CLI 用法

```bash
python cli.py --say "我住在京都"
python cli.py --dream            # 在种子数据后执行 1 次 Dreaming 整理
python cli.py --dream 3 --inspect  # 对前 3 个聚类执行整理并转储 DB
```

默认时区为 `Asia/Tokyo`（可通过 `MEMORY_TZ` 环境变量覆盖）。  
结果写入 `data/results.json`。

---

## 项目结构

```
├── core/               # Embedding、LLM 客户端、存储、指标、基础协议
├── memory/             # LongTermMemory 实现（检索、遗忘、Dreaming）
├── eval/               # 确定性保真度基准（Mock + 场景）
├── frontend/           # 无构建 React/HTM 单页应用
├── config.py           # GlobalConfig + LongTermMemoryConfig 数据类
├── server.py           # FastAPI 后端（REST API + 静态文件）
├── cli.py              # 无头运行器
├── seed_utterances.py  # 默认多年种子场景
├── requirements.txt
└── .env.example        # secrets/.env 模板
```

---

## 许可证

[MIT](LICENSE)

---

## 其他语言

- [English](README.md)
- [日本語](README.ja.md)
