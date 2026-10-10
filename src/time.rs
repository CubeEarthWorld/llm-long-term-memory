//! Clocks and the timezone field: `'IANA;+HH:MM'` is stamped once (the offset is
//! frozen into the field), then formatted with pure arithmetic valid for any year.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// The engine's time source: Unix seconds (fractional allowed).
pub type Clock = Box<dyn Fn() -> f64 + Send + Sync>;

/// The wall clock.
pub fn system_clock() -> Clock {
    Box::new(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()))
}

/// A settable clock shared by handle: tests, benchmarks and seed replays drive
/// virtual time through it.
#[derive(Clone, Debug, Default)]
pub struct ManualClock(Arc<AtomicU64>);

impl ManualClock {
    /// A clock reading `t` seconds.
    pub fn new(t: f64) -> Self {
        Self(Arc::new(AtomicU64::new(t.to_bits())))
    }
    /// The current reading.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::SeqCst))
    }
    /// Sets the reading.
    pub fn set(&self, t: f64) {
        self.0.store(t.to_bits(), Ordering::SeqCst);
    }
    /// Moves the reading forward by `seconds`.
    pub fn advance(&self, seconds: f64) {
        self.set(self.get() + seconds);
    }
    /// A [`Clock`] reading this handle.
    pub fn clock(&self) -> Clock {
        let me = self.clone();
        Box::new(move || me.get())
    }
}

/// `'IANA;+HH:MM'` for timezone `name` at `now` (the offset in force then, DST
/// included). An unknown zone falls back to `+00:00`.
pub fn tz_field(name: &str, now: f64) -> String {
    let offset = jiff::tz::TimeZone::get(name).ok().and_then(|tz| {
        let (lo, hi) = (jiff::Timestamp::MIN.as_second(), jiff::Timestamp::MAX.as_second());
        let ts = jiff::Timestamp::from_second((now as i64).clamp(lo, hi)).ok()?;
        Some(tz.to_offset(ts).seconds())
    });
    format!("{name};{}", fmt_offset(i64::from(offset.unwrap_or(0))))
}

fn fmt_offset(seconds: i64) -> String {
    let a = seconds.abs();
    format!("{}{:02}:{:02}", if seconds < 0 { '-' } else { '+' }, a / 3600, a % 3600 / 60)
}

/// `'+HH:MM'` / `'-HHMM'` / `'+HH'` → seconds; 0 when malformed.
pub fn parse_offset(text: &str) -> i64 {
    let t = text.trim();
    let (sign, rest) = match t.chars().next() {
        Some('+') => (1, &t[1..]),
        Some('-') => (-1, &t[1..]),
        _ => return 0,
    };
    let (h, m) = match rest.split_once(':') {
        Some(hm) => hm,
        None if (3..=4).contains(&rest.len()) => rest.split_at(rest.len() - 2),
        None => (rest, ""),
    };
    let digits = |s: &str, lo: usize, hi: usize| {
        (lo..=hi).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit())
    };
    if !digits(h, 1, 2) || !(m.is_empty() || digits(m, 2, 2)) {
        return 0;
    }
    sign * (h.parse::<i64>().unwrap_or(0) * 3600 + m.parse::<i64>().unwrap_or(0) * 60)
}

/// `'2026-06-11 21:30 +09:00'` from Unix seconds and a stored `'IANA;+offset'`
/// field: pure offset arithmetic (no tz database), valid for any year (SPEC §2).
pub fn fmt_local(unix: i64, tz_field: &str) -> String {
    let seconds = parse_offset(tz_field.split(';').nth(1).unwrap_or("+00:00"));
    let local = unix + seconds;
    let (days, sod) = (local.div_euclid(86_400), local.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} {}", sod / 3600, sod % 3600 / 60, fmt_offset(seconds))
}

/// Days since 1970-01-01 → proleptic Gregorian (year, month, day), any year.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_local_any_year_and_offsets() {
        assert_eq!(fmt_local(1_749_641_400, "Asia/Tokyo;+09:00"), "2025-06-11 20:30 +09:00");
        assert_eq!(fmt_local(0, "UTC;-03:30"), "1969-12-31 20:30 -03:30");
        assert!(fmt_local(253_402_300_800, "UTC;+00:00").starts_with("10000-01-01 00:00"));
        assert!(fmt_local(100_000_000_000, "X;+00:00").starts_with("5138-11-16"));
        assert_eq!(parse_offset("-0530"), -(5 * 3600 + 30 * 60));
        assert_eq!(parse_offset("+09"), 9 * 3600);
        assert_eq!(parse_offset("+9:00"), 9 * 3600);
        assert_eq!(parse_offset("garbage"), 0);
        assert_eq!(parse_offset("+09:"), 9 * 3600);
        assert_eq!(parse_offset("+12345"), 0);
    }

    #[test]
    fn tz_field_follows_dst_and_falls_back() {
        assert_eq!(tz_field("Asia/Tokyo", 1.7e9), "Asia/Tokyo;+09:00");
        assert_eq!(tz_field("America/New_York", 1_719_000_000.0), "America/New_York;-04:00"); // June: EDT
        assert_eq!(tz_field("America/New_York", 1_705_000_000.0), "America/New_York;-05:00"); // January: EST
        assert_eq!(tz_field("Nowhere/Land", 0.0), "Nowhere/Land;+00:00");
    }
}
