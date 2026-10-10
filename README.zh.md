# LLM Long-Term Memory

为 LLM 提供长期记忆的引擎，实现 **ENGRAM v2.2**（[`SPEC.md`](SPEC.md)，日文）：一个源自人类记忆原理的**记忆痕迹（trace）模型**。软件包是纯 Rust 的 crate `engram`。引擎不保存记忆的副本，需要什么就向存储查询，因此打开即用，每个操作只是几次索引查询。内置的 `Store` 把一切放在一个文件（redb）中，并带有会自动细分的相似检索索引；实现 `Storage` trait 即可换成任意数据库或向量索引。嵌入模型和 LLM 由使用者提供。没有网络功能、没有配置文件，依赖只有 `serde_json`、`jiff`、`ulid`，以及内置存储使用的 `redb`。

English: [README.md](README.md)　日本語: [README.ja.md](README.ja.md)

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
| `remember(text, salience, cue)` | 完全相同的文本视为复述；否则插入，**从不覆盖**。`cue` 触及更早近邻（cos ≥ θ_related）的痕迹生来*不稳定*，不改动已有痕迹。超过容量时，遗忘 3 天宽限期之外 strength 最小的痕迹（通过按稳定度分段的索引找出，不对全部排序）。 |
| `recall(query)` | 多线索余弦 → `score = a·(α + (1−α)·R)` → 绝对与相对阈值 → MMR → 以 `[unix tz] text 《id》` 注入 ≤1024 字。注入只是"暴露"，只强化一半。`peek(query)` 是无副作用的同样检索。 |
| `cite(reply)` | LLM 引用的《id》按"已使用"完全强化。 |
| `forget(id)` | 按 id 物理删除。 |
| `dream(adjudicator, budget)` | 以不稳定痕迹（先进先出，≤ 8·budget）为种子，聚合其 `cue` 重新激活的**更早**痕迹（cos ≥ θ_related，≤8），由 LLM 判定 **keep** 或 **replace**。要旨继承最强成员的稳定度加上其余成员的"活证据"；无关输出作为虚构被拒绝。 |

没有层级、计数器、环形缓冲、维护调用或快照复制。打开时不加载任何数据，也没有触及全部痕迹的操作：每个操作只是几次存储查询（有索引的存储为 O(log N)），且与经过时间无关；测试中包含 3000 虚拟年的模拟。

## 使用库

```rust
use engram::{Config, DreamDecision, DreamRequest, Engram, Store};

let mut memory = Engram::open(Box::new(Store::open("memory.redb".as_ref())?), Box::new(my_embedder), Config::default(), "Asia/Shanghai")?;
memory.remember("用户于2026年4月搬到了大阪。", 1.0, "用户住在哪里？")?;
let pack = memory.recall("我现在住在哪里？")?.pack_text;           // 注入到提示词
memory.cite(&llm_reply)?;                                           // 强化被使用的记忆
memory.dream(&mut |req: &DreamRequest| {                            // 任意能返回 JSON 的 LLM
    DreamDecision::parse(&my_llm(engram::DREAM_SYSTEM_PROMPT, &req.prompt())).ok_or("unparsable".into())
}, None)?;
```

接入点：`Embedder`（文本 → 向量）、`Storage`（持久化与检索：`find(view)`、`count`、`near` 相似检索、原子 `write`；内置 `Store` = 一个 redb 文件 + IVF 索引，用 `default-features = false` 可去掉）、`Adjudicator`（梦的判定，任意闭包）、`Clock`（时间）、`IdGen`（痕迹 id，默认 `ulid_ids()` 单调 ULID）。

## 许可证

[MIT](LICENSE)
