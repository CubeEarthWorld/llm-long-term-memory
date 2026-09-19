# LLM Long-Term Memory

为 LLM 提供长期记忆的层，实现 **ENGRAM v2**（[`SPEC.md`](SPEC.md)，日文）：一个源自人类记忆原理的**记忆痕迹（trace）模型**，仅依赖**本地嵌入模型**（EmbeddingGemma GGUF，经 llama.cpp 运行）和**单文件 SQLite**。LLM 只在三个时刻生成：写入、使用（并引用）、以及离线的**梦**（整合）。

> 只在言语化的瞬间生成。所有判断皆为距离。所有遗忘皆为算术。所有整合皆在梦中。

同一算法也以 Dart 包提供（[`../long-term-memory`](../long-term-memory)），并有跨语言一致性测试：同一脚本场景在两种语言中必须产生相同的执行轨迹。

## 模型概览

记忆是**痕迹**，只有两个数——上次被回忆的时刻与稳定度（半衰期）——加一个标志（`consolidated`）。

```text
R(now)   = 2^(−max(0, now − last_recall) / stability)          可提取性 ∈ [0,1]
strength = stability · R                                       未来可提取性总量
a        = max(0, (cos − cosine_floor) / (1 − cosine_floor))   线索激活
回忆      : stability ← min(stability · (1 + gain·a·(1−R)), S_max);  last_recall ← now
新痕迹    : stability = clamp(S0 · salience, 1 s, S_max)
```

| 动词 | 行为 |
|---|---|
| `remember(text, salience)` | 完全相同的文本视为复述；否则插入，**从不覆盖**。有邻居（cos ≥ θ_related）的痕迹以*不稳定*状态诞生并使邻居也不稳定（再巩固）。超出容量时，遗忘宽限期（3 天）之外强度最低的痕迹。 |
| `recall(query)` | 多线索余弦 → `score = a·(α + (1−α)·R)` → 绝对/相对阈值 → MMR → 以 `[unix tz] text 《id》` 注入 ≤1024 字。注入是"暴露"，只强化一半。 |
| `cite(reply)` | LLM 引用的《id》记忆按"使用"完整强化。 |
| `forget(id)` | 按 id 物理删除。 |
| `dream(budget)` | 不稳定痕迹（按稳定度）作为种子，取 cos ≥ θ_related 的邻居簇（≤8），由你的 LLM 判定 **keep** 或 **replace [texts]**。要旨继承最强成员的稳定度加其他成员的"活证据"；与输入无关的输出被视为虚构而拒绝。已整理的存储不会调用 LLM。 |

没有层级、计数器、环形队列或维护调用。全部状态有界，计算成本与经过的时间无关；测试套件包含 3000 虚拟年的模拟。

## 快速开始

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
mkdir secrets && copy .env.example secrets\.env       # 填写 DEEPSEEK_API_KEY（或 GEMINI_API_KEY）
# 将 embeddinggemma-300m-qat-Q4_0.gguf 放入 ./model（https://ai.google.dev/gemma/docs/embeddinggemma）
start.bat  /  ./start.sh                              # → http://localhost:8501
python cli.py --seed --dream 5 --inspect              # 命令行
python -m pytest                                      # 确定性测试（无需模型与密钥，含与 Dart 的一致性测试）；-m slow 为 3000 虚拟年模拟
```

参数（19 个）见 `config.py` 的 `LongTermMemoryConfig` 与 [`SPEC.md`](SPEC.md) §6。许可证：[MIT](LICENSE)。
