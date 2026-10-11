//! Trace ids: monotonic ULIDs (the `ulid` crate) by default, or any [`IdGen`].

use std::time::{Duration, UNIX_EPOCH};

/// Makes the id of a new trace from its Unix time in milliseconds and its text. Ids must
/// be unique, sort in creation order (a store loads by id, and the engine breaks ties
/// on insertion order) and contain no `《》`. The engine asks again while an id is taken.
pub type IdGen = Box<dyn FnMut(i64, &str) -> String + Send>;

/// The most a ULID's 48-bit timestamp holds (year 10,889).
const ULID_MAX_MILLIS: i64 = (1 << 48) - 1;

/// The default [`IdGen`]: ULIDs from one monotonic generator, so ids made in the same
/// millisecond still sort in creation order.
pub fn ulid_ids() -> IdGen {
    let mut generator = ulid::Generator::new();
    Box::new(move |millis, _| {
        let at = UNIX_EPOCH + Duration::from_millis(millis.clamp(0, ULID_MAX_MILLIS) as u64);
        generator.generate_from_datetime(at).unwrap_or_else(|overflow| overflow.commit_overflow_increment()).to_string()
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn ulids_sort_in_creation_order_within_a_millisecond() {
        let mut ids = super::ulid_ids();
        let same: Vec<String> = (0..100).map(|_| ids(5, "")).collect();
        assert!(same.iter().all(|id| id.len() == 26) && same.windows(2).all(|w| w[0] < w[1]));
        assert!(ids(6, "") > same[99]);
        assert_eq!(ids(i64::MAX, "").len(), 26); // a clock far out of range still makes an id
    }
}
