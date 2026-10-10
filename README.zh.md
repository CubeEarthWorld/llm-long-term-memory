# LLM Long-Term Memory

为 LLM 提供长期记忆的层，实现 **ENGRAM v2.1**（[`SPEC.md`](SPEC.md)，日文）：一个源自人类记忆原理的**记忆痕迹（trace）模型**，仅依赖**本地嵌入模型**（EmbeddingGemma GGUF，经 llama.cpp 运行；有 CUDA 时使用 GPU）和**单文件 SQLite**。以 Rust 编写：可嵌入智能体框架的轻依赖库（`engram`），以及应用 `ltm`（Web UI、无界面运行、基准测试）。LLM 只在三个时刻生成：写入、使用（并引用）、以及离线的**梦**（整合）。

同一算法也以 Dart 包提供（[`../long-term-memory`](../long-term-memory)）。两者共享同样的运算（float32 向量，余弦按维度顺序以 f64 累加），跨语言一致性场景与 220 步种子负载都能**逐位**复现 Dart 的执行轨迹。

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
| `remember(text, salience, cue)` | 完全相同的文本视为复述；否则插入，**从不覆盖**。`cue` 触及邻居（cos ≥ θ_related）的痕迹以*不稳定*状态诞生。超出容量时，遗忘宽限期（3 天）之外强度最低的痕迹。 |
| `recall(query)` | 多线索余弦 → `score = a·(α + (1−α)·R)` → 绝对/相对阈值 → MMR → 以 `[unix tz] text 《id》` 注入 ≤1024 字。注入是"暴露"，只强化一半。`peek(query)` 为无副作用的同一检索。 |
| `cite(reply)` | LLM 引用的《id》记忆按"使用"完整强化。 |
| `forget(id)` | 按 id 物理删除。 |
| `dream(adjudicator, budget)` | 不稳定痕迹（先进先出，≤ 8·budget）作为种子，取其 `cue` 所再激活的**更早**痕迹构成簇（cos ≥ θ_related，≤8），由 LLM 判定 **keep** 或 **replace**。要旨继承最强成员的稳定度加其他成员的"活证据"；无关输出被视为虚构而拒绝。 |

没有层级、计数器、环形队列或维护调用。每个操作至多对存储做一次线性扫描（按 id、按文本的查找为 O(1)，淘汰与邻居选取用部分选择而非全量排序），计算成本与经过的时间无关；测试包含 3000 虚拟年的模拟。

## 快速开始

需要 [Rust](https://rustup.rs) 以及带 CMake 和 libclang 的 C/C++ 工具链（Windows 上安装含 C++ 工作负载的 Visual Studio Build Tools 即可）。安装了 CUDA Toolkit 时，嵌入模型会自动放到 GPU 上。

```bash
mkdir secrets && copy .env.example secrets\.env       # LLM 默认经 Claude Code 调用 Claude（无需密钥）
# 将 embeddinggemma-300m-qat-Q4_0.gguf 放入 ./model（https://ai.google.dev/gemma/docs/embeddinggemma）
start.bat  /  ./start.sh                              # → http://localhost:8501
target/release/ltm run --dream 5 --inspect            # 命令行
cargo test                                            # 确定性测试（无需模型与密钥，含与 Dart 的一致性测试）
```

编译、llama.cpp 的 CPU 线程与基准测试的并行度均不超过 CPU 核心的 80%。参数（19 个）见 `engram::Config` 与 [`SPEC.md`](SPEC.md) §6。许可证：[MIT](LICENSE)。
