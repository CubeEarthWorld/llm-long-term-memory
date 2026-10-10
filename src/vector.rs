//! Brute-force cosine over unit vectors. Every score is the float32 inputs widened to
//! f64 and summed in dimension order — the Dart engine's arithmetic — so scores, and
//! therefore tie orders, agree bit for bit across implementations.

/// L2-normalised copy of `v` in f32 (accumulated in f64); a zero vector is returned as-is.
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|&x| (f64::from(x) / norm) as f32).collect()
}

/// Dot product (= cosine for unit vectors). Panics on a length mismatch: one model id
/// means one dimension, and a truncated product would be a cosine no threshold means.
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    a.iter().zip(b).fold(0.0, |s, (&x, &y)| s + f64::from(x) * f64::from(y))
}

/// Rows summed side by side: eight independent accumulator chains.
const LANES: usize = 8;
/// From this many queries on, widening each row block once pays for itself.
const BLOCKED_FROM: usize = 4;

/// Row-major scores: `out[i * queries.len() + j] == dot(rows[i], queries[j])`.
///
/// Rows are summed eight at a time in independent chains; with several queries each
/// block of rows is first widened into a dimension-major buffer and reused for every
/// query. Every chain still sums its own row in dimension order, so each score is
/// bit-identical to [`dot`].
pub fn scores(rows: &[&[f32]], queries: &[&[f32]]) -> Vec<f64> {
    let k = queries.len();
    let mut out = vec![0.0; rows.len() * k];
    let Some(n) = queries.first().map(|q| q.len()) else { return out };
    for r in rows.iter().map(|r| r.len()).chain(queries.iter().map(|q| q.len())) {
        assert_eq!(r, n, "vector dimension mismatch");
    }
    let qs: Vec<Vec<f64>> = queries.iter().map(|q| q.iter().map(|&x| f64::from(x)).collect()).collect();
    if k < BLOCKED_FROM {
        for (j, q) in qs.iter().enumerate() {
            for (b, block) in rows.chunks(LANES).enumerate() {
                let mut acc = [0.0f64; LANES];
                if let Ok(r) = <&[&[f32]; LANES]>::try_from(block) {
                    for (d, &x) in q.iter().enumerate() {
                        for (a, row) in acc.iter_mut().zip(r) {
                            *a += f64::from(row[d]) * x;
                        }
                    }
                } else {
                    for (a, row) in acc.iter_mut().zip(block) {
                        *a = row.iter().zip(q).fold(0.0, |s, (&v, &x)| s + f64::from(v) * x);
                    }
                }
                for (r, &s) in acc.iter().take(block.len()).enumerate() {
                    out[(b * LANES + r) * k + j] = s;
                }
            }
        }
        return out;
    }
    let mut buf = vec![[0.0f64; LANES]; n];
    for (b, block) in rows.chunks(LANES).enumerate() {
        for (r, row) in block.iter().enumerate() {
            for (lane, &x) in buf.iter_mut().zip(row.iter()) {
                lane[r] = f64::from(x);
            }
        }
        for (j, q) in qs.iter().enumerate() {
            let mut acc = [0.0f64; LANES];
            for (lane, &x) in buf.iter().zip(q) {
                for r in 0..LANES {
                    acc[r] += lane[r] * x;
                }
            }
            for (r, &s) in acc.iter().take(block.len()).enumerate() {
                out[(b * LANES + r) * k + j] = s;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_scores_equal_dot_bit_for_bit() {
        let rows: Vec<Vec<f32>> = (0..19).map(|i| (0..37).map(|d| ((i * 31 + d * 7) % 13) as f32 / 7.0 - 0.9).collect()).collect();
        let r: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        for k in [1, 3, 5] {
            // Both kernels (direct below BLOCKED_FROM queries, blocked from it on).
            let qs: Vec<Vec<f32>> = (0..k).map(|j| normalized(&rows[j * 2])).collect();
            let q: Vec<&[f32]> = qs.iter().map(Vec::as_slice).collect();
            let s = scores(&r, &q);
            for i in 0..rows.len() {
                for j in 0..k {
                    assert_eq!(s[i * k + j].to_bits(), dot(&rows[i], &qs[j]).to_bits());
                }
            }
        }
        assert_eq!(normalized(&[0.0, 0.0]), [0.0, 0.0]);
    }
}
