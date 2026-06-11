const html = htm.bind(React.createElement);
const { useState, useEffect, useCallback } = React;

async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

const apiPost = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

/* ----------------------------------------------------------------
   Inline functional UI glyphs (Lucide-style, 1.75 stroke).
   Not brand imagery — small affordances for nav & actions. Kept
   inline so the offline/no-build frontend needs no icon CDN.
   ---------------------------------------------------------------- */
const ICON_PATHS = {
  chat: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  list: "M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01",
  database: "M4 6c0 1.66 3.58 3 8 3s8-1.34 8-3-3.58-3-8-3-8 1.34-8 3zM4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6",
  chart: "M3 3v18h18M7 15v-4M12 15V8M17 15v-7",
  settings: "M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6",
  send: "M22 2 11 13M22 2l-7 20-4-9-9-4z",
  moon: "M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z",
  plus: "M12 5v14M5 12h14",
  refresh: "M21 2v6h-6M3 12a9 9 0 0 1 15-6.7L21 8M3 22v-6h6M21 12a9 9 0 0 1-15 6.7L3 16",
  sparkle: "M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17l-1.9-5.1L4.5 10l5.6-1.4z",
  doc: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6",
};

function Icon({ name, size }) {
  const d = ICON_PATHS[name];
  if (!d) return null;
  const s = size || 18;
  return html`<svg class="icon" width=${s} height=${s} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
    <path d=${d} /></svg>`;
}

function Badge({ kind, children }) {
  return html`<span class=${"badge " + kind}>${children}</span>`;
}

function DataTable({ rows }) {
  if (!rows || rows.length === 0) return html`<div class="empty-cell">データなし</div>`;
  const cols = [];
  rows.forEach((r) => Object.keys(r).forEach((k) => { if (!cols.includes(k)) cols.push(k); }));
  const fmt = (v) => {
    if (v === null || v === undefined) return "";
    let s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return s.length > 90 ? s.slice(0, 90) + "..." : s;
  };
  return html`
    <div class="tablewrap"><table>
      <thead><tr>${cols.map((c) => html`<th key=${c}>${c}</th>`)}</tr></thead>
      <tbody>${rows.map((r, i) => html`
        <tr key=${i}>${cols.map((c) => html`<td key=${c} title=${r[c] == null ? "" : String(r[c])}>${fmt(r[c])}</td>`)}</tr>`)}
      </tbody>
    </table></div>`;
}

function SystemCard({ detail }) {
  if (!detail || !detail.title) return html`<div class="answer-block"><div class="empty-cell">データなし</div></div>`;
  const t = detail.times;
  return html`
    <div class="answer-block">
      <div class="answer">${detail.response || "(空)"}</div>
      <div class="metarow">
        <span class="metachip">records ${detail.records}</span>
        <span class="metachip">pack ${detail.pack_chars}字 / ${detail.pack_n}件</span>
        <span class="metachip strong">total ${t.total}ms</span>
        <span class="metachip ghost">llm ${t.llm} · retr ${t.retrieve} · maint ${t.maintain} · embed ${t.embed}</span>
      </div>
      <div class="section-label">想起された記憶（READ・過去文脈として注入）</div>
      <${DataTable} rows=${detail.recalled} />
      <div class="section-label">save_memory / delete_memory の結果</div>
      <${DataTable} rows=${detail.written} />
      <details class="reveal">
        <summary><${Icon} name="doc" size=${14} /> 送信プロンプト / memory pack</summary>
        <pre class="pack">${detail.pack_text || "(empty)"}</pre>
        <pre class="pack">${(detail.prompt || "").slice(0, 4000)}</pre>
      </details>
    </div>`;
}

function fmtDate(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
function fmtElapsed(start, current) {
  if (!start || !current) return "";
  let sec = Math.max(0, Math.floor(current - start));
  const days = Math.floor(sec / 86400); sec %= 86400;
  const hours = Math.floor(sec / 3600); sec %= 3600;
  const mins = Math.floor(sec / 60); sec %= 60;
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours) parts.push(`${hours}h`);
  if (mins) parts.push(`${mins}m`);
  if (!parts.length) parts.push(`${sec}s`);
  return `+${parts.join(" ")}`;
}

function TurnBlock({ t, startTime }) {
  return html`
    <div class="turncard">
      <div class="turnhead">
        <span class="turnno">${t.turn}</span>
        <span class="turnutt">${t.utterance}</span>
        ${t.timestamp && html`<span class="turnclock">${fmtDate(t.timestamp)} <span class="turnelapsed">${fmtElapsed(startTime, t.timestamp)}</span></span>`}
        ${t.note && html`<span class="pill">${t.note}</span>`}
      </div>
      <${SystemCard} detail=${t.system} />
    </div>`;
}

function DreamPanel({ state, busy }) {
  const [results, setResults] = useState([]);
  const [err, setErr] = useState(null);
  const [msg, setMsg] = useState(null);
  const [open, setOpen] = useState(true);
  const load = useCallback(async () => {
    try { const d = await api("/dream-log"); setResults(d.results || []); } catch (e) {}
  }, []);
  useEffect(() => { load(); }, [state.turn, state.running, load]);
  const dream = async () => {
    setErr(null);
    setMsg(null);
    try {
      const res = await apiPost("/dream", { max_clusters: 5 });
      if (res.n === 0) {
        setMsg(res.message || "Dream 対象の適格クラスタがありません");
      }
    } catch (e) { setErr(e.message); }
  };
  const DREAM_LABEL = { merge: "統合", split: "分割", none: "変更なし" };
  return html`
    <div class="dream">
      <div class="dream-head">
        <div class="dream-title"><${Icon} name="moon" size=${17} /> Dream — 記憶の整理（破壊的操作はここだけ）</div>
        <div class="dream-actions">
          <button class="btn primary sm" disabled=${busy} onClick=${dream}>💤 Dream を実行</button>
          ${results.length > 0 && html`<button class="btn ghost sm" onClick=${() => setOpen(!open)}>${open ? "ログを隠す" : `ログ ${results.length}件`}</button>`}
        </div>
      </div>
      <div class="dream-hint">クラスタ化 → 審理 → 統合/分割。LLM が要点(gist)へまとめ、統合元は物理削除されます（1審理=1トランザクション）。</div>
      ${err && html`<div class="err">${err}</div>`}
      ${msg && html`<div class="kpi ok block-msg">${msg}</div>`}
      ${open && (results.length === 0 ? html`<div class="note">まだ Dream のログがありません。</div>` :
        results.map((r, i) => html`
          <div class="dreamcard" key=${i}>
            <div class="metarow">
              <span class="metachip">cluster ${r.cluster_fp || ""}</span>
              <span class=${"metachip action " + r.action}>${DREAM_LABEL[r.action] || r.action}</span>
              <span class="metachip ghost">priority ${r.priority}</span>
            </div>
            <div class="dreamcols">
              <div class="dream-col">
                <div class="col-label before">統合元 · ${r.before.length}</div>
                <ul class="mem-list">${r.before.map((b, j) => html`<li key=${j}><span class="w">L${b.tier} g${b.gen}</span>${b.text}</li>`)}</ul>
              </div>
              <div class="dream-col">
                <div class="col-label after">統合後 · ${r.after.length}</div>
                ${r.after.length ? html`<ul class="mem-list">${r.after.map((a, j) => html`<li key=${j}><span class="w">g${a.gen}</span>${a.text}</li>`)}</ul>` : html`<div class="note">変更なし（維持）</div>`}
              </div>
            </div>
          </div>`))}
    </div>`;
}

function ConversationView({ state, busy }) {
  const [turns, setTurns] = useState([]);
  const [startTime, setStartTime] = useState(null);
  const [sel, setSel] = useState("all");
  const [seedUtts, setSeedUtts] = useState([]);
  const [text, setText] = useState("");
  const [err, setErr] = useState(null);
  const [resetOnSeed, setResetOnSeed] = useState(true);
  const [seeding, setSeeding] = useState(false);

  const loadTurns = useCallback(async () => {
    const d = await api("/turns-detail");
    setTurns(d.turns || []);
    setStartTime(d.start_time || null);
  }, []);

  useEffect(() => { api("/seed-utterances").then((d) => setSeedUtts(d.utterances)).catch(() => {}); }, [state.turn]);
  useEffect(() => { loadTurns(); }, [state.turn, state.running, loadTurns]);
  // Release the local "seeding" latch once the server job has finished, so the
  // progress banner reflects real backend state even across a page reload.
  useEffect(() => { if (!state.running) setSeeding(false); }, [state.running]);

  const runSeed = async () => {
    setErr(null);
    setSeeding(true);  // immediate feedback — don't wait for the next 1.5s poll
    try { await apiPost("/seed", { reset: resetOnSeed }); }
    catch (e) { setErr(e.message); setSeeding(false); }
  };
  const inProgress = seeding || state.running;
  const sendTurn = async () => {
    if (!text.trim()) return;
    setErr(null);
    try { await apiPost("/turn", { text }); setText(""); } catch (e) { setErr(e.message); }
  };

  const shown = sel === "all" ? turns : turns.filter((t) => t.turn === Number(sel));
  return html`
    <div class="conv">
      <div class="toolbar">
        <button class="btn primary" disabled=${busy} onClick=${runSeed}>
          <${Icon} name="sparkle" size=${16} /> シード実行
        </button>
        <label class="checkline" title="チェック時は、シード投入前に記憶ストア（DB）を全消去してからやり直します。">
          <input type="checkbox" checked=${resetOnSeed} disabled=${busy}
            onChange=${(e) => setResetOnSeed(e.target.checked)} />
          <span>DBをリセット</span>
        </label>
        ${inProgress && html`<span class="kpi busy"><span class="spin"></span>${state.progress || "シード投入中..."}</span>`}
        ${state.seeded && !inProgress && html`<span class="kpi ok">シード済み · turn ${state.turn}</span>`}
        ${turns.length > 0 && html`
          <div class="spacer"></div>
          <label class="select-label">表示</label>
          <select class="select" value=${sel} onChange=${(e) => setSel(e.target.value)}>
            <option value="all">すべて（${turns.length}ターン）</option>
            ${turns.map((t) => html`<option key=${t.turn} value=${t.turn}>turn ${t.turn}: ${t.utterance.slice(0, 22)}</option>`)}
          </select>`}
      </div>
      ${err && html`<div class="err">${err}</div>`}
      ${inProgress && html`
        <div class="seedprogress">
          <span class="spin"></span>
          <span>${state.progress || "処理中..."}<span class="seedprogress-sub">　完了までしばらくお待ちください（シードは ${seedUtts.length} 発話を順に投入します）。</span></span>
        </div>`}

      ${turns.length === 0 && html`
        <div class="empty">
          <div class="empty-icon"><${Icon} name="sparkle" size=${26} /></div>
          <div class="empty-title">初期シード ${seedUtts.length} 件を投入して開始</div>
          <div class="empty-sub">「シード実行」を押すと、下記の発話が記憶として書き込まれます。</div>
          <div class="seedchips">${seedUtts.map((u) => html`
            <span class="seedchip" key=${u.i}><b>${u.i}</b>${u.text}${u.advance && u.advance !== "0" && html`<span class="pill">+${u.advance}</span>`}${u.note && html`<span class="pill">${u.note}</span>`}</span>`)}</div>
        </div>`}

      ${turns.length > 0 && html`
        <div class="turnstack">
          ${shown.map((t) => html`<${TurnBlock} key=${t.turn} t=${t} startTime=${startTime} />`)}
        </div>`}

      ${turns.length > 0 && html`
        <div class="composer">
          <input placeholder="次のターンの発話を入力..." value=${text} disabled=${busy}
            onKeyDown=${(e) => e.key === "Enter" && sendTurn()} onChange=${(e) => setText(e.target.value)} />
          <button class="btn primary" disabled=${busy || !text.trim()} onClick=${sendTurn}>
            <${Icon} name="send" size=${16} /> 送信
          </button>
        </div>`}

      <${DreamPanel} state=${state} busy=${busy} />
    </div>`;
}

function DBView() {
  const [data, setData] = useState(null);
  useEffect(() => { setData(null); api("/db").then(setData).catch(() => setData(null)); }, []);
  return html`
    <div>
      ${!data && html`<div class="note">読み込み中...</div>`}
      ${data && html`
        <div class="statgrid">${Object.entries(data.stats).map(([k, v]) =>
          html`<div class="statcard" key=${k}><div class="stat-num">${v}</div><div class="stat-label">${k}</div></div>`)}</div>
        ${Object.entries(data.tables).map(([tbl, rows]) => html`
          <div class="tblsection" key=${tbl}>
            <h2 class="tbl-title">${tbl}<span class="pill">${rows.length}</span></h2>
            <${DataTable} rows=${rows} />
          </div>`)}` }
    </div>`;
}

function LineChart({ rows, value, title }) {
  const W = 480, H = 190, pad = 34;
  const pts = rows.map((r) => ({ x: r.turn, y: r[value] }));
  if (!pts.length) return null;
  const xmin = Math.min(...pts.map((p) => p.x)), xmax = Math.max(...pts.map((p) => p.x), xmin + 1);
  const ymax = Math.max(...pts.map((p) => p.y), 1), ymin = 0;
  const sx = (x) => pad + (x - xmin) / (xmax - xmin || 1) * (W - 2 * pad);
  const sy = (y) => H - pad - (y - ymin) / (ymax - ymin || 1) * (H - 2 * pad);
  const line = pts.map((pt) => sx(pt.x) + "," + sy(pt.y)).join(" ");
  const area = `${sx(pts[0].x)},${sy(ymin)} ${line} ${sx(pts[pts.length - 1].x)},${sy(ymin)}`;
  return html`
    <div class="chart-card">
      <div class="chart-title">${title}</div>
      <svg width=${W} height=${H} viewBox=${"0 0 " + W + " " + H} class="chart-svg">
        <defs><linearGradient id=${"g-" + value} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.18" />
          <stop offset="100%" stop-color="var(--accent)" stop-opacity="0" />
        </linearGradient></defs>
        <line x1=${pad} y1=${H - pad} x2=${W - pad} y2=${H - pad} stroke="var(--line)" />
        <line x1=${pad} y1=${pad} x2=${pad} y2=${H - pad} stroke="var(--line)" />
        <text x=${pad - 6} y=${pad} font-size="10" fill="var(--ink-3)" text-anchor="end">${ymax}</text>
        <text x=${pad - 6} y=${H - pad} font-size="10" fill="var(--ink-3)" text-anchor="end">0</text>
        <polygon fill=${"url(#g-" + value + ")"} points=${area} />
        <polyline fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round" points=${line} />
        ${pts.map((pt, i) => html`<circle key=${i} cx=${sx(pt.x)} cy=${sy(pt.y)} r="3" fill="#fff" stroke="var(--accent)" stroke-width="2" />`)}
      </svg>
    </div>`;
}

function MetricsView({ state }) {
  const [data, setData] = useState(null);
  useEffect(() => { api("/metrics").then(setData).catch(() => {}); }, [state.turn, state.running]);
  if (!data || !data.rows.length) return html`<div class="note">まだ実行データがありません。</div>`;
  return html`
    <div>
      <div class="charts">
        <${LineChart} rows=${data.rows} value="records" title="総レコード数の推移" />
        <${LineChart} rows=${data.rows} value="vector_mb" title="ベクトルデータ量 (MB)" />
      </div>
      <div class="invariants">${Object.entries(data.invariants).map(([k, v]) =>
        html`<div key=${k} class=${"inv " + (v ? "ok" : "ng")}><span class="inv-tag">${v ? "OK" : "NG"}</span>${k}</div>`)}</div>
      <h2 class="tbl-title">生メトリクス</h2>
      <${DataTable} rows=${data.rows} />
    </div>`;
}

const LABELS = {
  // global
  llm_provider: "LLMプロバイダー", deepseek_model: "Deepseekモデル", gemini_model: "Geminiモデル",
  deepseek_base_url: "Deepseek Base URL", temperature: "temperature", max_output_tokens: "最大出力token",
  embedding_model: "埋め込みモデル", dim_full: "L1ベクトル次元(768)", budget_chars: "注入予算(字)",
  default_timezone: "既定タイムゾーン", max_turn_log: "ターンログ上限", max_metrics_history: "メトリクス履歴上限",
  // ENGRAM §8 parameters
  cap1: "L1容量(エピソード)", cap2: "L2容量(意味)", cap3: "L3容量(スキーマ)",
  dim1: "L1次元 768 f32", dim2: "L2次元 256 int8", dim3: "L3次元 128 int8",
  tau1: "L1半減期τ(秒/7日)", tau2: "L2半減期τ(秒/90日)", tau3: "L3半減期τ(秒/3年)",
  m_max: "mass上限 M_max", refractory_seconds: "不応期(秒)",
  alpha: "活性の床α(スコア)", inject_n: "注入件数", mmr_lambda: "MMR多様性λ",
  theta_same: "同一閾値θ_same", theta_conflict: "競合閾値θ_conflict", precise_margin: "精査マージン",
  theta_up: "昇格閾値θ_up(A)", theta_down: "降格閾値θ_down(A)",
  text_max: "記憶本文上限(字)", text_hard_max: "本文ハード上限(B)", gen_max: "統合世代上限",
  dream_budget: "夢の審理数 budget", dream_budget_hard: "夢budgetハード上限",
  cluster_min: "クラスタ適格(最小件数)", dream_max_members: "1審理の最大メンバ", snapshot_gens: "スナップショット世代",
  conflict_cap: "conflictリング容量", dream_log_cap: "dream_logリング容量",
  tombstone_sweep_pct: "墓標掃除(層比率)", tombstone_sweep_age: "墓標掃除(経過秒)",
  write_rate_per_day: "書込みレート/日", max_writes_per_turn: "1ターン保存上限",
  tool_fallback: "保存フォールバック(抽出)",
  hard_memory_rows: "memory総行ハード上限", hard_vec_rows: "vec総行ハード上限", decay_exp_cap: "減衰指数上限(A=0)",
};

const GLOB_GROUPS = [
  ["LLM", ["llm_provider", "deepseek_model", "gemini_model", "deepseek_base_url", "temperature", "max_output_tokens"]],
  ["埋め込み・注入", ["embedding_model", "dim_full", "budget_chars", "default_timezone"]],
];

function SettingsView({ state, busy, onApplied }) {
  const [cfg, setCfg] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { api("/config").then(setCfg).catch(() => {}); }, [state.ready]);
  if (!cfg) return html`<div class="note">設定を読み込み中...</div>`;

  const setField = (sec, key, val) => setCfg((c) => ({ ...c, [sec]: { ...c[sec], [key]: val } }));
  const apply = async () => {
    setErr(null);
    try { await apiPost("/reset", { config: cfg }); onApplied(); } catch (e) { setErr(e.message); }
  };
  const resetDb = async () => {
    setErr(null);
    try { await apiPost("/reset-db"); onApplied(); } catch (e) { setErr(e.message); }
  };

  const field = (sec, key, val) => {
    const id = sec + "." + key;
    const label = LABELS[key] || key;
    if (sec === "glob" && key === "llm_provider")
      return html`<div class="field" key=${id}><label>${label}</label>
        <div class="seg">${["deepseek", "gemini"].map((o) => html`
          <button key=${o} class=${"segbtn " + (val === o ? "on" : "")} onClick=${() => setField(sec, key, o)}>${o}</button>`)}
        </div></div>`;
    if (typeof val === "boolean")
      return html`<div class="field" key=${id}><label title=${id}>${label}</label>
        <div class="seg">${[["on", true], ["off", false]].map(([lbl, o]) => html`
          <button key=${lbl} class=${"segbtn " + (val === o ? "on" : "")} onClick=${() => setField(sec, key, o)}>${lbl}</button>`)}
        </div></div>`;
    if (typeof val === "number")
      return html`<div class="field" key=${id}><label title=${id}>${label}</label>
        <input type="number" step="any" value=${val} onChange=${(e) => setField(sec, key, e.target.value === "" ? 0 : Number(e.target.value))} /></div>`;
    return html`<div class="field" key=${id}><label title=${id}>${label}</label>
      <input type="text" value=${val} onChange=${(e) => setField(sec, key, e.target.value)} /></div>`;
  };

  return html`
    <div class="settings">
      ${err && html`<div class="err">${err}</div>`}
      <${ExplanationBox} />
      <div class="group-card">
        <div class="group-card-head">グローバル設定</div>
        <div class="group-card-body">
          ${GLOB_GROUPS.map(([title, keys]) => html`
            <div key=${title} class="subgroup">
              <div class="subhead">${title}</div>
              <div class="fieldgrid">${keys.filter((k) => k in cfg.glob).map((k) => field("glob", k, cfg.glob[k]))}</div>
            </div>`)}
        </div>
      </div>
      <div class="group-card">
        <div class="group-card-head">ENGRAM v1.1 パラメータ（§8）</div>
        <div class="group-card-body">
          <div class="fieldgrid">${Object.entries(cfg.memory).map(([k, v]) => field("memory", k, v))}</div>
        </div>
      </div>
      <div class="settings-actions">
        <button class="btn primary" disabled=${busy} onClick=${apply}>設定を適用してリセット</button>
        <button class="btn danger" disabled=${busy} onClick=${resetDb}>DBデータをリセット（設定は維持）</button>
      </div>
    </div>`;
}

function ExplanationBox() {
  return html`
    <details class="explain">
      <summary><${Icon} name="sparkle" size=${15} /> ENGRAM v1.1 概要</summary>
      <div class="explain-body">
        <p><b>生成は言語化の瞬間だけ。判断はすべて距離。忘却はすべて算術。破壊はすべて夢の中。</b></p>
        <p>記憶の本体は短い自己完結テキスト(≤170字)で、ベクトルは導出物(索引)です。記憶は3層: <b>L1 エピソード</b>(768d f32, τ=7日) / <b>L2 意味</b>(256d int8, τ=90日) / <b>L3 スキーマ</b>(128d int8, τ=3年)。MRL次元切詰め＝忘却の解像度です。</p>
        <p>活性 <code>A = mass · 2^(−Δt/τ)</code>。想起のたびに <code>mass</code> を +1(不応期1時間)。スコアは <code>max(0,cos) × (α + (1−α)·A_abs)</code>(α=0.35)、注入5件は MMR で多様化、≤1024字。</p>
        <p>同一性はコサイン距離のみ(文書—文書): <code>≥0.97</code> 同一更新(旧に墓標+新挿入=再固定化) / <code>0.85–0.97</code> 競合(両保持→conflictキュー) / <code>&lt;0.85</code> 新規。</p>
        <p>会話中は<b>追記のみ</b>(非破壊)。容量超過は機械移動: A≥16 で昇格(L→L1)、A&lt;4 で降格(次元を粗く)、L3 溢れは活性最下位から物理削除(=この系の死)。</p>
        <p>💤 Dream(オフライン)だけが破壊的: クラスタ化→LLM審理(統合/分割/変更なし)→統合元を物理削除。1審理=1トランザクション、スナップショット8世代で保護。</p>
        <p>DBは自己記述: <code>spec</code> テーブルに本仕様全文を同梱。依存は「単一ファイルDB・埋め込みモデル・算術」のみ。</p>
      </div>
    </details>`;
}

function csvCell(s) {
  s = s == null ? "" : String(s);
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function SeedEditor({ busy, onChanged }) {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState(null);
  const [msg, setMsg] = useState(null);

  const load = useCallback(async () => {
    const d = await api("/seed-utterances");
    setItems((d.utterances || []).map((u) => ({ text: u.text, note: u.note || "", advance: u.advance || "0" })));
  }, []);
  useEffect(() => { load().catch((e) => setErr(e.message)); }, [load]);
  if (!items) return html`<div class="note">読み込み中...</div>`;

  const setRow = (i, key, val) => setItems((a) => a.map((r, j) => (j === i ? { ...r, [key]: val } : r)));
  const addRow = () => setItems((a) => [...a, { text: "", note: "", advance: "0" }]);
  const delRow = (i) => setItems((a) => a.filter((_, j) => j !== i));
  const move = (i, d) => setItems((a) => {
    const j = i + d;
    if (j < 0 || j >= a.length) return a;
    const b = a.slice(); const t = b[i]; b[i] = b[j]; b[j] = t; return b;
  });
  const clean = () => items.map((r) => ({ text: r.text.trim(), note: (r.note || "").trim(), advance: ((r.advance || "0").trim() || "0") })).filter((r) => r.text);

  const save = async (alsoSeed) => {
    setErr(null); setMsg(null);
    const payload = clean();
    if (!payload.length) { setErr("少なくとも1件の発話が必要です。"); return; }
    try {
      await apiPost("/seed-utterances", { items: payload });
      setItems(payload);
      onChanged && onChanged();
      if (alsoSeed) { await apiPost("/seed"); setMsg("保存し、シードを実行しました（会話タブで確認）。"); }
    } catch (e) { setErr(e.message); }
  };
  const restore = async () => {
    setErr(null); setMsg(null);
    try { const d = await apiPost("/seed-utterances/reset"); await load(); onChanged && onChanged(); setMsg(`既定の${d.n}件に戻しました。`); }
    catch (e) { setErr(e.message); }
  };
  const exportCsv = () => {
    const csv = "\\ufeff" + ["text,note,advance", ...items.map((r) => csvCell(r.text) + "," + csvCell(r.note) + "," + csvCell(r.advance || "0"))].join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "seed.csv"; document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
  };
  const importCsv = async (e) => {
    setErr(null); setMsg(null);
    const f = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const txt = await f.text();
      const d = await apiPost("/seed-utterances/import", { csv: txt });
      setItems(d.items.map((r) => ({ text: r.text, note: r.note || "", advance: r.advance || "0" })));
    } catch (err) { setErr(err.message); }
  };

  return html`
    <div>
      ${err && html`<div class="err">${err}</div>`}
      ${msg && html`<div class="kpi ok block-msg">${msg}</div>`}
      <div class="seed-toolbar">
        <button class="btn primary" disabled=${busy} onClick=${() => save(false)}>保存</button>
        <button class="btn" disabled=${busy} onClick=${() => save(true)}>保存して実行</button>
        <div class="spacer"></div>
        <button class="btn ghost" disabled=${busy} onClick=${addRow}><${Icon} name="plus" size=${15} /> 行を追加</button>
        <button class="btn ghost" disabled=${busy} onClick=${exportCsv}>CSV書き出し</button>
        <label class="btn ghost filelbl">CSV読み込み
          <input type="file" accept=".csv,text/csv" style=${{ display: "none" }} disabled=${busy} onChange=${importCsv} />
        </label>
        <button class="btn ghost" disabled=${busy} onClick=${restore}><${Icon} name="refresh" size=${15} /> 既定に戻す</button>
        <span class="kpi">${items.length}件</span>
      </div>
      <div class="seedtable">
        <table>
          <thead><tr><th style=${{ width: 40 }}>#</th><th>発話テキスト</th><th style=${{ width: 200 }}>メモ（任意）</th><th style=${{ width: 110 }}>仮想時間</th><th style=${{ width: 120 }}>操作</th></tr></thead>
          <tbody>
            ${items.map((r, i) => html`
              <tr key=${i}>
                <td class="rownum">${i + 1}</td>
                <td><input class="seedinput" value=${r.text} disabled=${busy}
                      onChange=${(e) => setRow(i, "text", e.target.value)} placeholder="発話を入力..." /></td>
                <td><input class="seedinput" value=${r.note} disabled=${busy}
                      onChange=${(e) => setRow(i, "note", e.target.value)} placeholder="ノイズ / 重要 など" /></td>
                <td><input class="seedinput" value=${r.advance || "0"} disabled=${busy}
                      onChange=${(e) => setRow(i, "advance", e.target.value)} title="直前の経過時間を進める量（例: 0, 12h, 8d, 5y）" placeholder="例: 8d, 5y" /></td>
                <td class="seedops">
                  <button class="iconbtn" disabled=${busy || i === 0} onClick=${() => move(i, -1)} title="上へ">↑</button>
                  <button class="iconbtn" disabled=${busy || i === items.length - 1} onClick=${() => move(i, 1)} title="下へ">↓</button>
                  <button class="iconbtn del" disabled=${busy} onClick=${() => delRow(i)} title="削除">×</button>
                </td>
              </tr>`)}
          </tbody>
        </table>
      </div>
    </div>`;
}

function StatusChip({ label, value }) {
  if (!value) return null;
  const ok = value.startsWith("OK");
  return html`<div class=${"hchip " + (ok ? "ok" : "err")} title=${value}>
    <span class="hdot"></span><span class="hchip-label">${label}</span>
    <span class="hchip-val">${value.replace(/^OK\s*/, "")}</span>
  </div>`;
}

const NAV = [
  ["chat", "会話", "chat"],
  ["seed", "シード編集", "list"],
  ["db", "DB閲覧", "database"],
  ["metrics", "メトリクス", "chart"],
  ["settings", "設定", "settings"],
];

function App() {
  const [state, setState] = useState(null);
  const [tab, setTab] = useState("chat");

  const poll = useCallback(async () => {
    try { setState(await api("/state")); } catch (e) {}
  }, []);
  useEffect(() => {
    poll();
    const interval = state && state.running ? 800 : 1500;
    const id = setInterval(poll, interval);
    return () => clearInterval(id);
  }, [poll, state && state.running]);
  if (!state) return html`<div class="boot"><span class="spin"></span> 起動中...</div>`;

  const busy = state.running || !state.ready;
  const titles = { chat: "会話", seed: "シード編集", db: "DB閲覧", metrics: "メトリクス", settings: "設定" };
  const descs = {
    chat: "シードを投入し、ターンごとに想起・書き込み・Dream を確認します。",
    seed: "初期記憶として投入する発話を編集します。CSV 入出力に対応。",
    db: "現在の記憶ストア（SQLite）の中身を一覧します。",
    metrics: "実行ごとのレコード数・ベクトル量と不変条件をモニタします。",
    settings: "LLM・埋め込み・記憶パラメータの調整と、システムのリセット。",
  };

  return html`
    <div class="app">
      <header class="topbar">
        <div class="brand">
          <div class="brand-mark"><${Icon} name="sparkle" size=${18} /></div>
          <div class="brand-text">
            <h1>LLM Long-Term Memory</h1>
            <span class="brand-sub">ENGRAM v1.1 — EmbeddingGemma（ローカル） + Deepseek / Gemini</span>
          </div>
        </div>
        <div class="topstatus">
          <${StatusChip} label="埋め込み" value=${state.embedding} />
          <${StatusChip} label="LLM" value=${state.llm} />
          <div class="turnchip">turn <b>${state.turn}</b></div>
          ${busy && html`<div class="hchip busy"><span class="spin"></span>${state.progress || "running"}</div>`}
        </div>
      </header>

      ${(state.init_error || state.error) && html`
        <div class="alertbar">${state.init_error || state.error}</div>`}

      <div class="shell">
        <nav class="nav">
          ${NAV.map(([k, label, icon]) => html`
            <button key=${k} class=${"navitem " + (k === tab ? "active" : "")} onClick=${() => setTab(k)}>
              <span class="navicon"><${Icon} name=${icon} size=${18} /></span>${label}
            </button>`)}
        </nav>

        <main class="content">
          <div class="viewhead">
            <h2 class="view-title">${titles[tab]}</h2>
            <p class="view-desc">${descs[tab]}</p>
          </div>
          <div class="view">
            ${tab === "chat" && html`<${ConversationView} state=${state} busy=${busy} />`}
            ${tab === "seed" && html`<${SeedEditor} busy=${busy} onChanged=${poll} />`}
            ${tab === "db" && html`<${DBView} />`}
            ${tab === "metrics" && html`<${MetricsView} state=${state} />`}
            ${tab === "settings" && html`<${SettingsView} state=${state} busy=${busy} onApplied=${poll} />`}
          </div>
        </main>
      </div>
    </div>`;
}

ReactDOM.createRoot(document.getElementById("root")).render(html`<${App} />`);
