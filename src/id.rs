//! ULID-shaped trace ids.

use std::hash::{BuildHasher, Hasher, RandomState};
use std::sync::atomic::{AtomicU64, Ordering};

const CROCKFORD32: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Seed of reproducible ids (0 = random ids).
static SEED: AtomicU64 = AtomicU64::new(0);

/// Makes every following trace id a function of `seed`, the trace's text and its
/// timestamp, so a replayed scenario writes the same ids — and therefore sends the same
/// prompts — every run, independently of what happened before. `0` restores random ids.
pub fn deterministic_ids(seed: u64) {
    SEED.store(seed, Ordering::SeqCst);
}

fn splitmix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// 64 random bits from the standard library's per-process random keys, a counter and
/// the clock — ids need uniqueness, not cryptographic strength.
fn random_u64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut h = RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    h.finish()
}

fn encode(millis: i64, mut a: u64, mut b: u64) -> String {
    let mut ts = millis.max(0) as u64;
    let mut out = [0u8; 26];
    for slot in out[..10].iter_mut().rev() {
        *slot = CROCKFORD32[(ts % 32) as usize];
        ts /= 32;
    }
    for (i, slot) in out[10..].iter_mut().enumerate() {
        let r = if i < 8 { &mut a } else { &mut b };
        *slot = CROCKFORD32[(*r & 31) as usize];
        *r >>= 5;
    }
    String::from_utf8(out.to_vec()).expect("ASCII")
}

/// 26-char ULID-shaped id: 10 chars = 50-bit millisecond timestamp (valid past year
/// 37,000), 16 chars = 80 random bits. Lexicographic order = time order.
pub fn ulid(millis: i64) -> String {
    encode(millis, random_u64(), random_u64())
}

/// The id of a new trace: random, or — after [`deterministic_ids`] — derived from the
/// seed, the text and the timestamp.
pub(crate) fn trace_id(millis: i64, text: &str) -> String {
    let seed = SEED.load(Ordering::SeqCst);
    if seed == 0 {
        return ulid(millis);
    }
    let h = text.bytes().fold(0xCBF2_9CE4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3));
    let a = splitmix(seed ^ h ^ (millis as u64).rotate_left(32));
    encode(millis, a, splitmix(a))
}

#[cfg(test)]
mod tests {
    use super::{trace_id, ulid};

    #[test]
    fn sorts_by_time_and_survives_50_bits() {
        let (a, b) = (ulid(1_700_000_000_000), ulid(1_700_000_000_001));
        assert!(a.len() == 26 && a < b);
        let far = ulid(1 << 49);
        assert!(far > b && !far.starts_with('0'));
        assert_ne!(ulid(5), ulid(5));
        assert_ne!(trace_id(5, "x"), trace_id(5, "x")); // random until deterministic_ids
    }
}
