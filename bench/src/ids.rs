//! Reproducible trace ids for scenario replays: a function of a seed, the trace's text
//! and its timestamp, so a replay writes the same ids — and therefore sends the same
//! prompts — every run, independently of what happened before.

use engram::IdGen;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// ULIDs whose random part is drawn from `seed`, the text, the millisecond and a salt.
/// The engine asks again while an id is taken (a gist that repeats its seed's text in
/// the same millisecond); a repeated request gets the next salt.
pub fn reproducible(seed: u64) -> IdGen {
    let mut last: Option<(i64, String, u64)> = None;
    Box::new(move |millis, text| {
        let salt = match &last {
            Some((m, t, s)) if *m == millis && t == text => s + 1,
            _ => 0,
        };
        last = Some((millis, text.to_owned(), salt));
        // FNV-1a: a fixed hash, so the ids do not change with the toolchain.
        let bytes = [&seed.to_le_bytes()[..], &millis.to_le_bytes(), &salt.to_le_bytes(), text.as_bytes()].concat();
        let h = bytes.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3));
        let random: u128 = StdRng::seed_from_u64(h).random();
        ulid::Ulid::from_parts(millis.max(0) as u64, random).to_string()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn same_inputs_same_ids_and_a_repeat_gets_a_new_salt() {
        let (mut a, mut b) = (super::reproducible(7), super::reproducible(7));
        let x = a(5, "t");
        assert_eq!(x, b(5, "t"));
        assert_ne!(a(5, "t"), x); // asked again: taken
        assert_eq!(a(6, "u"), b(6, "u"));
    }
}
