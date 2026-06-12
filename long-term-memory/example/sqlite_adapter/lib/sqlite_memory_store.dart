/// Example [MemoryStore] adapter over `package:sqlite3`.
///
/// Demonstrates everything a production adapter should provide: a schema
/// mirroring the ENGRAM reference layout, WAL mode, real transactions for
/// dream adjudications, a rotating snapshot ring for [backup], and
/// checkpoint+vacuum for [compact]. On Flutter, add `sqlite3_flutter_libs`
/// and this file works unchanged.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:long_term_memory/long_term_memory.dart';
import 'package:sqlite3/sqlite3.dart';

const _schema = '''
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  tier INTEGER NOT NULL,
  gen INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  tz TEXT NOT NULL,
  last_access INTEGER NOT NULL,
  last_bonus_at INTEGER NOT NULL DEFAULT 0,
  mass REAL NOT NULL DEFAULT 1.0,
  superseded_by TEXT,
  CHECK (tier IN (1,2,3)),
  CHECK (length(text) BETWEEN 1 AND 1024)
);
CREATE TABLE IF NOT EXISTS vec (
  memory_id TEXT NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,
  dim INTEGER NOT NULL,
  dtype TEXT NOT NULL,
  scale REAL,
  v BLOB NOT NULL,
  PRIMARY KEY (memory_id, model_id)
);
CREATE TABLE IF NOT EXISTS conflict (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  a TEXT, b TEXT, at INTEGER
);
CREATE TABLE IF NOT EXISTS dream_log (fp TEXT PRIMARY KEY, verdict TEXT, at INTEGER);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(tier);
CREATE INDEX IF NOT EXISTS idx_memory_super ON memory(superseded_by);
CREATE INDEX IF NOT EXISTS idx_memory_text ON memory(text);
''';

/// [MemoryStore] backed by a single SQLite file (or `:memory:`).
class SqliteMemoryStore extends MemoryStore {
  /// Creates a store at [path] (`':memory:'` for an in-memory database).
  /// [snapshotGenerations] is the size of the rotating [backup] ring.
  SqliteMemoryStore(this.path, {this.snapshotGenerations = 8});

  /// Database file path.
  final String path;

  /// Number of rotating snapshot files kept by [backup].
  final int snapshotGenerations;

  late Database _db;
  var _open = false;
  var _inTransaction = false;

  @override
  Future<void> initialize() async {
    if (_open) return;
    _db = sqlite3.open(path);
    _db.execute('PRAGMA journal_mode=WAL;');
    _db.execute('PRAGMA auto_vacuum=INCREMENTAL;');
    _db.execute('PRAGMA foreign_keys=ON;');
    _db.execute(_schema);
    _open = true;
  }

  @override
  Future<void> close() async {
    if (!_open) return;
    _db.dispose();
    _open = false;
  }

  // ------------------------------------------------------------------ //
  // memory rows
  // ------------------------------------------------------------------ //

  MemoryRecord _row(Row r) => MemoryRecord(
        id: r['id'] as String,
        text: r['text'] as String,
        tier: r['tier'] as int,
        gen: r['gen'] as int,
        createdAt: r['created_at'] as int,
        tz: r['tz'] as String,
        lastAccess: r['last_access'] as int,
        lastBonusAt: r['last_bonus_at'] as int,
        mass: (r['mass'] as num).toDouble(),
        supersededBy: r['superseded_by'] as String?,
      );

  (String, List<Object?>) _where(int? tier, Liveness liveness) {
    final conds = <String>[];
    final args = <Object?>[];
    if (tier != null) {
      conds.add('tier = ?');
      args.add(tier);
    }
    switch (liveness) {
      case Liveness.liveOnly:
        conds.add('superseded_by IS NULL');
      case Liveness.tombstonesOnly:
        conds.add('superseded_by IS NOT NULL');
      case Liveness.all:
        break;
    }
    return (conds.isEmpty ? '' : ' WHERE ${conds.join(' AND ')}', args);
  }

  @override
  Future<void> insertMemory(MemoryRecord m) async {
    _db.execute(
      'INSERT INTO memory(id,text,tier,gen,created_at,tz,last_access,'
      'last_bonus_at,mass,superseded_by) VALUES(?,?,?,?,?,?,?,?,?,?)',
      [
        m.id, m.text, m.tier, m.gen, m.createdAt, m.tz, //
        m.lastAccess, m.lastBonusAt, m.mass, m.supersededBy,
      ],
    );
  }

  @override
  Future<MemoryRecord?> getMemory(String id) async {
    final rows = _db.select('SELECT * FROM memory WHERE id=?', [id]);
    return rows.isEmpty ? null : _row(rows.first);
  }

  @override
  Future<MemoryRecord?> getLiveMemoryByText(String text) async {
    final rows = _db.select(
        'SELECT * FROM memory WHERE text=? AND superseded_by IS NULL LIMIT 1',
        [text]);
    return rows.isEmpty ? null : _row(rows.first);
  }

  @override
  Future<List<MemoryRecord>> listMemories({
    int? tier,
    Liveness liveness = Liveness.all,
  }) async {
    final (where, args) = _where(tier, liveness);
    return _db
        .select('SELECT * FROM memory$where', args)
        .map(_row)
        .toList(growable: false);
  }

  @override
  Future<int> countMemories({
    int? tier,
    Liveness liveness = Liveness.all,
  }) async {
    final (where, args) = _where(tier, liveness);
    return _db.select('SELECT COUNT(*) c FROM memory$where', args).first['c']
        as int;
  }

  @override
  Future<void> updateMemory(MemoryRecord m) async {
    _db.execute(
      'UPDATE memory SET text=?,tier=?,gen=?,created_at=?,tz=?,last_access=?,'
      'last_bonus_at=?,mass=?,superseded_by=? WHERE id=?',
      [
        m.text, m.tier, m.gen, m.createdAt, m.tz, m.lastAccess, //
        m.lastBonusAt, m.mass, m.supersededBy, m.id,
      ],
    );
  }

  @override
  Future<void> deleteMemory(String id) async {
    _db.execute('DELETE FROM memory WHERE id=?', [id]); // vec cascades
  }

  // ------------------------------------------------------------------ //
  // vectors
  // ------------------------------------------------------------------ //

  VectorRecord _vec(Row r) => VectorRecord(
        memoryId: r['memory_id'] as String,
        modelId: r['model_id'] as String,
        dim: r['dim'] as int,
        dtype: VecDtype.parse(r['dtype'] as String),
        scale: (r['scale'] as num?)?.toDouble(),
        bytes: r['v'] as Uint8List,
      );

  @override
  Future<void> upsertVector(VectorRecord v) async {
    _db.execute(
      'INSERT OR REPLACE INTO vec(memory_id,model_id,dim,dtype,scale,v) '
      'VALUES(?,?,?,?,?,?)',
      [v.memoryId, v.modelId, v.dim, v.dtype.label, v.scale, v.bytes],
    );
  }

  @override
  Future<VectorRecord?> getVector(String memoryId, String modelId) async {
    final rows = _db.select(
        'SELECT * FROM vec WHERE memory_id=? AND model_id=?',
        [memoryId, modelId]);
    return rows.isEmpty ? null : _vec(rows.first);
  }

  @override
  Future<List<VectorRecord>> listVectors({String? modelId}) async {
    final rows = modelId == null
        ? _db.select('SELECT * FROM vec')
        : _db.select('SELECT * FROM vec WHERE model_id=?', [modelId]);
    return rows.map(_vec).toList(growable: false);
  }

  @override
  Future<void> deleteVector(String memoryId, String modelId) async {
    _db.execute('DELETE FROM vec WHERE memory_id=? AND model_id=?',
        [memoryId, modelId]);
  }

  // ------------------------------------------------------------------ //
  // conflict ring / dream log / meta
  // ------------------------------------------------------------------ //

  @override
  Future<void> addConflict(ConflictRecord c) async {
    _db.execute('INSERT INTO conflict(a,b,at) VALUES(?,?,?)', [c.a, c.b, c.at]);
  }

  @override
  Future<List<ConflictRecord>> listConflicts() async => _db
      .select('SELECT a,b,at FROM conflict ORDER BY seq')
      .map((r) => ConflictRecord(
          a: r['a'] as String, b: r['b'] as String, at: r['at'] as int))
      .toList(growable: false);

  @override
  Future<int> countConflicts() async =>
      _db.select('SELECT COUNT(*) c FROM conflict').first['c'] as int;

  @override
  Future<void> trimConflicts(int maxEntries) async {
    _db.execute(
      'DELETE FROM conflict WHERE seq IN (SELECT seq FROM conflict '
      'ORDER BY seq ASC LIMIT max(0, (SELECT COUNT(*) FROM conflict) - ?))',
      [maxEntries],
    );
  }

  @override
  Future<String?> getDreamVerdict(String fingerprint) async {
    final rows =
        _db.select('SELECT verdict FROM dream_log WHERE fp=?', [fingerprint]);
    return rows.isEmpty ? null : rows.first['verdict'] as String?;
  }

  @override
  Future<void> putDreamVerdict(DreamLogRecord r) async {
    _db.execute('INSERT OR REPLACE INTO dream_log(fp,verdict,at) VALUES(?,?,?)',
        [r.fingerprint, r.verdict, r.at]);
  }

  @override
  Future<int> countDreamLog() async =>
      _db.select('SELECT COUNT(*) c FROM dream_log').first['c'] as int;

  @override
  Future<void> trimDreamLog(int maxEntries) async {
    _db.execute(
      'DELETE FROM dream_log WHERE fp IN (SELECT fp FROM dream_log '
      'ORDER BY at ASC LIMIT max(0, (SELECT COUNT(*) FROM dream_log) - ?))',
      [maxEntries],
    );
  }

  @override
  Future<String?> getMeta(String key) async {
    final rows = _db.select('SELECT v FROM meta WHERE k=?', [key]);
    return rows.isEmpty ? null : rows.first['v'] as String?;
  }

  @override
  Future<void> putMeta(String key, String value) async {
    _db.execute('INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)', [key, value]);
  }

  @override
  Future<void> clear() async {
    for (final t in const ['vec', 'memory', 'conflict', 'dream_log', 'meta']) {
      _db.execute('DELETE FROM $t');
    }
  }

  // ------------------------------------------------------------------ //
  // bulk overrides — one statement each
  // ------------------------------------------------------------------ //

  @override
  Future<void> clampFutureTimestamps(int nowUnix) async {
    _db.execute('UPDATE memory SET last_access=? WHERE last_access>?',
        [nowUnix, nowUnix]);
    _db.execute('UPDATE memory SET last_bonus_at=? WHERE last_bonus_at>?',
        [nowUnix, nowUnix]);
    _db.execute('UPDATE memory SET created_at=? WHERE created_at>?',
        [nowUnix, nowUnix]);
  }

  @override
  Future<void> deleteTombstones({int? tier, int? lastAccessBefore}) async {
    final conds = ['superseded_by IS NOT NULL'];
    final args = <Object?>[];
    if (tier != null) {
      conds.add('tier=?');
      args.add(tier);
    }
    if (lastAccessBefore != null) {
      conds.add('last_access<?');
      args.add(lastAccessBefore);
    }
    _db.execute('DELETE FROM memory WHERE ${conds.join(' AND ')}', args);
  }

  @override
  Future<void> deleteVectorsExceptModel(String modelId) async {
    _db.execute('DELETE FROM vec WHERE model_id<>?', [modelId]);
  }

  @override
  Future<void> deleteOrphanVectors() async {
    _db.execute(
        'DELETE FROM vec WHERE memory_id NOT IN (SELECT id FROM memory)');
  }

  @override
  Future<List<TierEntry>> listTierEntries(int tier, String modelId) async {
    final rows = _db.select(
      'SELECT m.*, v.memory_id, v.model_id, v.dim, v.dtype, v.scale, v.v '
      'FROM memory m JOIN vec v ON v.memory_id = m.id AND v.model_id = ? '
      'WHERE m.tier = ? AND m.superseded_by IS NULL',
      [modelId, tier],
    );
    return rows
        .map((r) => TierEntry(memory: _row(r), vector: _vec(r)))
        .toList(growable: false);
  }

  // ------------------------------------------------------------------ //
  // hooks
  // ------------------------------------------------------------------ //

  @override
  Future<T> runInTransaction<T>(Future<T> Function() action) async {
    if (_inTransaction) return action(); // engine never nests, but be safe
    _db.execute('BEGIN IMMEDIATE');
    _inTransaction = true;
    try {
      final result = await action();
      _db.execute('COMMIT');
      return result;
    } catch (_) {
      _db.execute('ROLLBACK');
      rethrow;
    } finally {
      _inTransaction = false;
    }
  }

  @override
  Future<void> backup() async {
    if (path == ':memory:') return;
    final dir = Directory('$path.snapshots');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    // Rotate: snap-(n-1) dropped, others shifted up, newest written to snap-0.
    final last = File('${dir.path}/snap-${snapshotGenerations - 1}.db');
    if (last.existsSync()) last.deleteSync();
    for (var i = snapshotGenerations - 2; i >= 0; i--) {
      final f = File('${dir.path}/snap-$i.db');
      if (f.existsSync()) f.renameSync('${dir.path}/snap-${i + 1}.db');
    }
    final target = '${dir.path}/snap-0.db';
    _db.execute('VACUUM INTO ?', [target]);
  }

  @override
  Future<void> compact() async {
    _db.execute('PRAGMA wal_checkpoint(TRUNCATE);');
    _db.execute('PRAGMA incremental_vacuum;');
  }

  @override
  Future<int> sizeInBytes() async {
    if (path == ':memory:') return -1;
    final f = File(path);
    return f.existsSync() ? f.lengthSync() : -1;
  }
}
