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
        <span class="metachip ghost">llm ${t.llm} · retr ${t.retrieve} · write ${t.write} · maint ${t.maintain}</span>
      </div>
      <div class="section-label">LLMが呼び出した記憶</div>
      <${DataTable} rows=${detail.recalled} />
      <div class="section-label">書き込まれた / 更新された記憶</div>
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
      const res = await apiPost("/dream", { max_clusters: 1 }); 
      if (res.n === 0) {
        setMsg(res.message || "Dream 対象のクラスタがありません");
      }
    } catch (e) { setErr(e.message); }
  };
  return html`
    <div class="dream">
      <div class="dream-head">
        <div class="dream-title"><${Icon} name="moon" size=${17} /> Dream — 記憶の整理</div>
        <div class="dream-actions">
          <button class="btn primary sm" disabled=${busy} onClick=${dream}>💤 Dream を実行</button>
          ${results.length > 0 && html`<button class="btn ghost sm" onClick=${() => setOpen(!open)}>${open ? "ログを隠す" : `ログ ${results.length}件`}</button>`}
        </div>
      </div>
      <div class="dream-hint">優先度の高いクラスタを LLM が 統合 / 分割 / 維持 します。</div>
      ${err && html`<div class="err">${err}</div>`}
      ${msg && html`<div class="kpi ok block-msg">${msg}</div>`}
      ${open && (results.length === 0 ? html`<div class="note">まだ Dream のログがありません。</div>` :
        results.map((r, i) => html`
          <div class="dreamcard" key=${i}>
            <div class="metarow">
              <span class="metachip">cluster ${String(r.cluster_id).slice(0, 8)}</span>
              <span class=${"metachip action " + r.action}>${r.action}</span>
              <span class="metachip ghost">priority ${r.priority}</span>
            </div>
            <div class="dreamcols">
              <div class="dream-col">
                <div class="col-label before">before · ${r.before.length}</div>
                <ul class="mem-list">${r.before.map((b, j) => html`<li key=${j}><span class="w">${b.w}</span>${b.text}</li>`)}</ul>
              </div>
              <div class="dream-col">
                <div class="col-label after">after · ${r.after.length}</div>
                ${r.after.length ? html`<ul class="mem-list">${r.after.map((a, j) => html`<li key=${j}><span class="w">${a.w}</span><span class="tzpill">${a.timezone}</span>${a.text}</li>`)}</ul>` : html`<div class="note">変更なし（維持）</div>`}
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

  const loadTurns = useCallback(async () => {
    const d = await api("/turns-detail");
    setTurns(d.turns || []);
    setStartTime(d.start_time || null);
  }, []);

  useEffect(() => { api("/seed-utterances").then((d) => setSeedUtts(d.utterances)).catch(() => {}); }, [state.turn]);
  useEffect(() => { loadTurns(); }, [state.turn, state.running, loadTurns]);

  const runSeed = async () => {
    setErr(null);
    try { await apiPost("/seed"); } catch (e) { setErr(e.message); }
  };
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
        ${state.seeded && html`<span class="kpi ok">シード済み · turn ${state.turn}</span>`}
        ${turns.length > 0 && html`
          <div class="spacer"></div>
          <label class="select-label">表示</label>
          <select class="select" value=${sel} onChange=${(e) => setSel(e.target.value)}>
            <option value="all">すべて（${turns.length}ターン）</option>
            ${turns.map((t) => html`<option key=${t.turn} value=${t.turn}>turn ${t.turn}: ${t.utterance.slice(0, 22)}</option>`)}
          </select>`}
      </div>
      ${err && html`<div class="err">${err}</div>`}

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
  llm_provider: "LLMプロバイダー", deepseek_model: "Deepseekモデル", gemini_model: "Geminiモデル",
  deepseek_base_url: "Deepseek Base URL", temperature: "temperature", max_output_tokens: "最大出力token",
  embedding_model: "埋め込みモデル", budget_chars: "取得最大文字数", total_cap: "記憶上限(hot)",
  archive_cap: "アーカイブ上限(cold)",
  sim_floor: "類似度しきい値", dim_full: "検索ベクトル次元", dim_coarse: "coarseベクトル次元",
  default_timezone: "既定タイムゾーン",
  mem_max_chars: "記憶1件の最大文字数", tau_dup: "重複判定", tau_link: "クラスタ判定",
  stab_base_seconds: "基準安定度S(秒)", kappa: "重要度→安定度係数",
  forget_beta: "忘却べき指数β", stab_growth_c: "想起時の安定度成長係数", labile_frac: "labile開始率",
  freq_seed: "初期アクセス回数", reinforce_inc: "アクセス時加算",
  min_residency_seconds: "最小滞在秒",
  confidence_user: "信頼度(user)", confidence_inferred: "信頼度(inferred)",
  alpha: "類似度重みα", beta: "保持率重みβ", delta: "重要度重みδ", eta: "頻度重みη", zeta: "信頼度重みζ",
  lambda_div: "多様性係数", k_retrieve: "検索候補件数", spread_gamma: "連想拡散係数γ",
  gate_w_cos: "想起ゲート:cos重み", gate_w_r: "想起ゲート:r重み", gate_theta: "想起ゲート閾値θ",
  recall_noise_sigma: "想起ノイズσ",
  r_archive_floor: "アーカイブ閾値r", r_hard_floor: "保護解除r下限",
  archive_grace_seconds: "アーカイブ猶予秒", tau_savings: "再出現判定(coarse cos)", savings_gain: "savings安定度ゲイン",
  tau_recall: "思い出し閾値(coarse cos)", max_recall_per_turn: "1回の復元上限(思い出し)",
  protect_confidence: "保護:信頼度下限", protect_w: "保護:重要度下限",
  interference_decay: "干渉減衰", consolidation_gain: "固定化ゲイン",
  dream_min_size: "Dream最小サイズ", dream_min_interval_seconds: "Dreamクールダウン秒",
  dream_max_members: "Dream最大メンバ", dream_max_clusters: "Dream対象クラスタ数",
  dream_priority_floor: "Dream優先度下限",
  dream_w_size: "優先度: サイズ重み", dream_w_spread: "優先度: 時間幅重み",
  dream_w_age: "優先度: 経過重み", dream_w_disp: "優先度: 非凝集重み",
  dream_spread_norm: "時間幅正規化(秒)", dream_age_norm: "経過正規化(秒)",
};

const GLOB_GROUPS = [
  ["LLM", ["llm_provider", "deepseek_model", "gemini_model", "deepseek_base_url", "temperature", "max_output_tokens"]],
  ["埋め込み・取得", ["embedding_model", "dim_full", "dim_coarse", "budget_chars", "total_cap", "archive_cap", "sim_floor", "default_timezone"]],
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
    if (sec === "glob" && key === "dim_coarse")
      return html`<div class="field" key=${id}><label>${label}</label>
        <select class="select" value=${val} onChange=${(e) => setField(sec, key, Number(e.target.value))}>
          <option value=${768}>768</option><option value=${256}>256</option></select></div>`;
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
        <div class="group-card-head">LLM Long-Term Memory パラメータ</div>
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
      <summary><${Icon} name="sparkle" size=${15} /> LLM Long-Term Memory 概要</summary>
      <div class="explain-body">
        <p>記憶は2層です: 想起対象の <b>active(hot, 上限1000件)</b> と、忘れて引けなくなった <b>archive(cold, 上限5000件, savings用)</b>。総容量は有界です。</p>
        <p>検索には768次元のfull embeddingを使い、256次元のcoarse（full から都度導出）はクラスタ・多様性・連想拡散・archive照合に使います。</p>
        <p>各記憶は人間の記憶に倣い3軸を分離: <code>w</code>(重要度) / <code>confidence</code>(信頼度) / <code>S</code>(定着度=安定度)。忘却はべき乗則 <code>r = (1 + (now-accessed_at)/S)^(-β)</code>。</p>
        <p>想起成功のたびに安定度が伸びます（忘れかけ＝r が低いほど大きく＝間隔/テスト効果）。<code>S ← S * (1 + c*(1-r))</code>、アクセス回数 <code>freq</code> も加算。</p>
        <p>想起ランキング: <code>α·cos + β·r + δ·w + η·log1p(freq) + ζ·confidence</code> ＋ 連想拡散。想起は <code>r</code> でゲートされ、引けない記憶は機能的に忘却されます。</p>
        <p>忘却: <code>r</code> が閾値を下回ると archive へ退避（引けなくなるが savings として復元可能）。同じ話題が再来すると安定度の先取りで復元。archive 超過分は永久削除。</p>
        <p>💤 Dream はクラスタ単位の睡眠的整理で、LLM が記憶を統合（要点=gist化）/ 分割 / 維持します。統合記憶は耐久性が増し、由来（source_ids）を残します。</p>
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
  useEffect(() => { poll(); const id = setInterval(poll, 1500); return () => clearInterval(id); }, [poll]);
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
            <span class="brand-sub">EmbeddingGemma（ローカル） + Deepseek / Gemini</span>
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
