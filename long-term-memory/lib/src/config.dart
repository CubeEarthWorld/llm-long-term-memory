/// ENGRAM v1.1 parameters (spec §8). All defaults match the reference
/// implementation; tune only with a record of why.
class EngramConfig {
  /// Creates a configuration. Every parameter has the spec default.
  const EngramConfig({
    this.cap1 = 1000,
    this.cap2 = 3000,
    this.cap3 = 6000,
    this.dim1 = 768,
    this.dim2 = 256,
    this.dim3 = 128,
    this.tau1 = 7 * _day,
    this.tau2 = 90 * _day,
    this.tau3 = 3 * _year,
    this.mMax = 64.0,
    this.refractorySeconds = 3600,
    this.alpha = 0.35,
    this.injectN = 5,
    this.mmrLambda = 0.3,
    this.scoreThresholds = const [0.1, 0.2],
    this.budgetChars = 1024,
    this.thetaSame = 0.97,
    this.thetaConflict = 0.85,
    this.preciseMargin = 0.03,
    this.thetaUp = 16.0,
    this.thetaDown = 4.0,
    this.textMax = 170,
    this.textHardMax = 1024,
    this.genMax = 7,
    this.dreamBudget = 5,
    this.dreamBudgetHard = 4096,
    this.clusterMin = 3,
    this.clusterCohesionMin = 0.5,
    this.dreamMaxMembers = 64,
    this.conflictCap = 256,
    this.dreamLogCap = 512,
    this.tombstoneSweepPct = 0.10,
    this.tombstoneSweepAge = 7 * _day,
    this.writeRatePerDay = 2000,
    this.hardMemoryRows = 16384,
    this.decayExpCap = 65536.0,
    this.compactEvery = 20,
  })  : assert(dim1 >= dim2 && dim2 >= dim3, 'MRL dims must be descending'),
        assert(cap1 + cap2 + cap3 <= hardMemoryRows,
            'tier capacities exceed hardMemoryRows'),
        assert(
            thetaSame > thetaConflict, 'theta_same must exceed theta_conflict'),
        assert(thetaUp > thetaDown,
            'promotion/demotion hysteresis requires θ_up > θ_down');

  static const double _day = 24 * 60 * 60;
  static const double _year = 365 * _day;

  // -- Tier capacities (tombstones included, I4) and MRL vector format -- //

  /// L1 (episodic) capacity in rows.
  final int cap1;

  /// L2 (semantic) capacity in rows.
  final int cap2;

  /// L3 (schema) capacity in rows.
  final int cap3;

  /// L1 vector dimension (float32). Also the engine's "full" precision dim:
  /// the embedder must produce at least this many dimensions.
  final int dim1;

  /// L2 vector dimension (int8, MRL truncation of the full vector).
  final int dim2;

  /// L3 vector dimension (int8).
  final int dim3;

  // -- Activation half-lives in seconds (§4.1): L1=7d, L2=90d, L3=3y -- //

  /// L1 half-life τ in seconds.
  final double tau1;

  /// L2 half-life τ in seconds.
  final double tau2;

  /// L3 half-life τ in seconds.
  final double tau3;

  /// Mass / activation upper bound (I1).
  final double mMax;

  /// Minimum interval between mass bonuses — the spacing-effect refractory
  /// period (§4.1).
  final double refractorySeconds;

  // -- Retrieval score (§4.2) -- //

  /// Activation floor in the retrieval score: dormant but relevant memories
  /// still compete.
  final double alpha;

  /// Number of memories injected per retrieve call.
  final int injectN;

  /// MMR diversity penalty λ.
  final double mmrLambda;

  /// Progressive score thresholds, relaxed (strictest first) only when
  /// nothing matches; if even the loosest matches nothing, inject nothing.
  final List<double> scoreThresholds;

  /// Character budget of the injected memory pack.
  final int budgetChars;

  // -- Identity thresholds (document–document cosine, §4.3) -- //

  /// cos ≥ this → same proposition: supersede the old row.
  final double thetaSame;

  /// cos in `[thetaConflict, thetaSame)` → conflict: keep both, queue for
  /// dream adjudication.
  final double thetaConflict;

  /// On write, L2/L3 candidates within this margin of [thetaConflict] are
  /// re-embedded at full precision before the verdict (§5.1).
  final double preciseMargin;

  // -- Tier movement hysteresis (§4.4) -- //

  /// Activation ≥ this promotes L2/L3 rows to L1 during dream.
  final double thetaUp;

  /// Activation < this marks a row as a preferred demotion victim.
  final double thetaDown;

  // -- Text atomicity (§3.1, §8) -- //

  /// Soft limit: maximum characters of one stored proposition (longer text
  /// is shortened at a sentence boundary).
  final int textMax;

  /// Hard limit in UTF-8 bytes; longer writes are rejected (I14).
  final int textHardMax;

  /// Consolidation generation cap (I7).
  final int genMax;

  // -- Dream (§6, §8) -- //

  /// Default LLM adjudications per `dream()` call.
  final int dreamBudget;

  /// Absolute bound on the dream budget ("∞" per §8.1).
  final int dreamBudgetHard;

  /// Minimum members for an eligible cluster.
  final int clusterMin;

  /// Minimum mean pairwise cosine for cluster eligibility.
  final double clusterCohesionMin;

  /// Maximum members handed to the LLM per adjudication.
  final int dreamMaxMembers;

  // -- Ring buffers (I11) -- //

  /// Conflict queue capacity (oldest dropped beyond this).
  final int conflictCap;

  /// Dream log capacity.
  final int dreamLogCap;

  // -- Housekeeping (§6) -- //

  /// Sweep a tier's tombstones when they exceed this fraction of the tier.
  final double tombstoneSweepPct;

  /// ...or when a tombstone is older than this many seconds.
  final double tombstoneSweepAge;

  /// Soft write-rate limit per rolling 24 h (§5.1).
  final int writeRatePerDay;

  // -- Safety hard bounds (§8.1, I14) -- //

  /// Hard cap on total memory rows — also the loop guard for capacity
  /// enforcement.
  final int hardMemoryRows;

  /// If `max(0,Δt)/τ` exceeds this, activation is exactly 0 (underflow guard).
  final double decayExpCap;

  /// Call `MemoryStore.compact()` every N `maintain()` calls.
  final int compactEvery;

  /// Half-life for [tier] (1-based).
  double tauOf(int tier) => switch (tier) { 1 => tau1, 2 => tau2, _ => tau3 };

  /// Vector dimension for [tier] (1-based).
  int dimOf(int tier) => switch (tier) { 1 => dim1, 2 => dim2, _ => dim3 };

  /// Capacity for [tier] (1-based).
  int capOf(int tier) => switch (tier) { 1 => cap1, 2 => cap2, _ => cap3 };

  /// Returns a copy with the given fields replaced.
  EngramConfig copyWith({
    int? cap1,
    int? cap2,
    int? cap3,
    int? dim1,
    int? dim2,
    int? dim3,
    double? tau1,
    double? tau2,
    double? tau3,
    double? mMax,
    double? refractorySeconds,
    double? alpha,
    int? injectN,
    double? mmrLambda,
    List<double>? scoreThresholds,
    int? budgetChars,
    double? thetaSame,
    double? thetaConflict,
    double? preciseMargin,
    double? thetaUp,
    double? thetaDown,
    int? textMax,
    int? textHardMax,
    int? genMax,
    int? dreamBudget,
    int? dreamBudgetHard,
    int? clusterMin,
    double? clusterCohesionMin,
    int? dreamMaxMembers,
    int? conflictCap,
    int? dreamLogCap,
    double? tombstoneSweepPct,
    double? tombstoneSweepAge,
    int? writeRatePerDay,
    int? hardMemoryRows,
    double? decayExpCap,
    int? compactEvery,
  }) {
    return EngramConfig(
      cap1: cap1 ?? this.cap1,
      cap2: cap2 ?? this.cap2,
      cap3: cap3 ?? this.cap3,
      dim1: dim1 ?? this.dim1,
      dim2: dim2 ?? this.dim2,
      dim3: dim3 ?? this.dim3,
      tau1: tau1 ?? this.tau1,
      tau2: tau2 ?? this.tau2,
      tau3: tau3 ?? this.tau3,
      mMax: mMax ?? this.mMax,
      refractorySeconds: refractorySeconds ?? this.refractorySeconds,
      alpha: alpha ?? this.alpha,
      injectN: injectN ?? this.injectN,
      mmrLambda: mmrLambda ?? this.mmrLambda,
      scoreThresholds: scoreThresholds ?? this.scoreThresholds,
      budgetChars: budgetChars ?? this.budgetChars,
      thetaSame: thetaSame ?? this.thetaSame,
      thetaConflict: thetaConflict ?? this.thetaConflict,
      preciseMargin: preciseMargin ?? this.preciseMargin,
      thetaUp: thetaUp ?? this.thetaUp,
      thetaDown: thetaDown ?? this.thetaDown,
      textMax: textMax ?? this.textMax,
      textHardMax: textHardMax ?? this.textHardMax,
      genMax: genMax ?? this.genMax,
      dreamBudget: dreamBudget ?? this.dreamBudget,
      dreamBudgetHard: dreamBudgetHard ?? this.dreamBudgetHard,
      clusterMin: clusterMin ?? this.clusterMin,
      clusterCohesionMin: clusterCohesionMin ?? this.clusterCohesionMin,
      dreamMaxMembers: dreamMaxMembers ?? this.dreamMaxMembers,
      conflictCap: conflictCap ?? this.conflictCap,
      dreamLogCap: dreamLogCap ?? this.dreamLogCap,
      tombstoneSweepPct: tombstoneSweepPct ?? this.tombstoneSweepPct,
      tombstoneSweepAge: tombstoneSweepAge ?? this.tombstoneSweepAge,
      writeRatePerDay: writeRatePerDay ?? this.writeRatePerDay,
      hardMemoryRows: hardMemoryRows ?? this.hardMemoryRows,
      decayExpCap: decayExpCap ?? this.decayExpCap,
      compactEvery: compactEvery ?? this.compactEvery,
    );
  }

  /// JSON form (for persisting / displaying the active configuration).
  Map<String, Object?> toJson() => {
        'cap1': cap1,
        'cap2': cap2,
        'cap3': cap3,
        'dim1': dim1,
        'dim2': dim2,
        'dim3': dim3,
        'tau1': tau1,
        'tau2': tau2,
        'tau3': tau3,
        'mMax': mMax,
        'refractorySeconds': refractorySeconds,
        'alpha': alpha,
        'injectN': injectN,
        'mmrLambda': mmrLambda,
        'scoreThresholds': scoreThresholds,
        'budgetChars': budgetChars,
        'thetaSame': thetaSame,
        'thetaConflict': thetaConflict,
        'preciseMargin': preciseMargin,
        'thetaUp': thetaUp,
        'thetaDown': thetaDown,
        'textMax': textMax,
        'textHardMax': textHardMax,
        'genMax': genMax,
        'dreamBudget': dreamBudget,
        'dreamBudgetHard': dreamBudgetHard,
        'clusterMin': clusterMin,
        'clusterCohesionMin': clusterCohesionMin,
        'dreamMaxMembers': dreamMaxMembers,
        'conflictCap': conflictCap,
        'dreamLogCap': dreamLogCap,
        'tombstoneSweepPct': tombstoneSweepPct,
        'tombstoneSweepAge': tombstoneSweepAge,
        'writeRatePerDay': writeRatePerDay,
        'hardMemoryRows': hardMemoryRows,
        'decayExpCap': decayExpCap,
        'compactEvery': compactEvery,
      };

  /// Lenient inverse of [toJson]: missing or mistyped values fall back to
  /// the defaults.
  factory EngramConfig.fromJson(Map<String, Object?> json) {
    const d = EngramConfig();
    int i(String k, int def) => switch (json[k]) {
          final num v => v.toInt(),
          final String v => int.tryParse(v) ?? def,
          _ => def,
        };
    double f(String k, double def) => switch (json[k]) {
          final num v => v.toDouble(),
          final String v => double.tryParse(v) ?? def,
          _ => def,
        };
    List<double> list(String k, List<double> def) => switch (json[k]) {
          final List<Object?> v =>
            v.whereType<num>().map((e) => e.toDouble()).toList(),
          _ => def,
        };
    return EngramConfig(
      cap1: i('cap1', d.cap1),
      cap2: i('cap2', d.cap2),
      cap3: i('cap3', d.cap3),
      dim1: i('dim1', d.dim1),
      dim2: i('dim2', d.dim2),
      dim3: i('dim3', d.dim3),
      tau1: f('tau1', d.tau1),
      tau2: f('tau2', d.tau2),
      tau3: f('tau3', d.tau3),
      mMax: f('mMax', d.mMax),
      refractorySeconds: f('refractorySeconds', d.refractorySeconds),
      alpha: f('alpha', d.alpha),
      injectN: i('injectN', d.injectN),
      mmrLambda: f('mmrLambda', d.mmrLambda),
      scoreThresholds: list('scoreThresholds', d.scoreThresholds),
      budgetChars: i('budgetChars', d.budgetChars),
      thetaSame: f('thetaSame', d.thetaSame),
      thetaConflict: f('thetaConflict', d.thetaConflict),
      preciseMargin: f('preciseMargin', d.preciseMargin),
      thetaUp: f('thetaUp', d.thetaUp),
      thetaDown: f('thetaDown', d.thetaDown),
      textMax: i('textMax', d.textMax),
      textHardMax: i('textHardMax', d.textHardMax),
      genMax: i('genMax', d.genMax),
      dreamBudget: i('dreamBudget', d.dreamBudget),
      dreamBudgetHard: i('dreamBudgetHard', d.dreamBudgetHard),
      clusterMin: i('clusterMin', d.clusterMin),
      clusterCohesionMin: f('clusterCohesionMin', d.clusterCohesionMin),
      dreamMaxMembers: i('dreamMaxMembers', d.dreamMaxMembers),
      conflictCap: i('conflictCap', d.conflictCap),
      dreamLogCap: i('dreamLogCap', d.dreamLogCap),
      tombstoneSweepPct: f('tombstoneSweepPct', d.tombstoneSweepPct),
      tombstoneSweepAge: f('tombstoneSweepAge', d.tombstoneSweepAge),
      writeRatePerDay: i('writeRatePerDay', d.writeRatePerDay),
      hardMemoryRows: i('hardMemoryRows', d.hardMemoryRows),
      decayExpCap: f('decayExpCap', d.decayExpCap),
      compactEvery: i('compactEvery', d.compactEvery),
    );
  }
}
