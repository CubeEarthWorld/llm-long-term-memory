//! Brute-force cosine over unit vectors — the engine's only heavy arithmetic.
//!
//! A dot product is multiply-adds in float32 over sixteen independent lanes, which the
//! compiler turns into SIMD; the lanes are summed in f64. For unit vectors of a few
//! thousand dimensions the error stays near 1e-7, far below any threshold the engine
//! compares against.

/// Independent accumulator lanes (two AVX2 registers of eight f32).
const LANES: usize = 16;

/// L2-normalised copy of `v`; a zero vector is returned as-is.
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = dot(v, v).sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
}

/// Dot product (= cosine for unit vectors). Panics on a length mismatch: one model id
/// means one dimension, and a truncated product would be a cosine no threshold means.
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for l in 0..LANES {
            acc[l] = x[l].mul_add(y[l], acc[l]);
        }
    }
    let tail: f64 = ar.iter().zip(br).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
    acc.iter().map(|&s| f64::from(s)).sum::<f64>() + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_the_exact_sum() {
        let a: Vec<f32> = (0..771).map(|d| ((d * 7) % 13) as f32 / 7.0 - 0.9).collect();
        let b: Vec<f32> = (0..771).map(|d| ((d * 5) % 11) as f32 / 5.0 - 1.1).collect();
        let exact: f64 = a.iter().zip(&b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
        assert!((dot(&a, &b) - exact).abs() < 1e-4 * exact.abs().max(1.0));
        let n = normalized(&a);
        assert!((dot(&n, &n) - 1.0).abs() < 1e-6);
        assert_eq!(normalized(&[0.0, 0.0]), [0.0, 0.0]);
    }
}
