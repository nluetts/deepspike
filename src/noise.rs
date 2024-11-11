use oorandom::Rand32;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn pseudo_normal(n: usize, seed: Option<u64>) -> Vec<f32> {
    let mut xs = Vec::with_capacity(n);
    let seed = seed.unwrap_or_else(time_ns);
    let mut rng = Rand32::new(seed);
    let mut num = 0.0;
    let mut count = 0;
    for _ in 0..n * 12 {
        num += rng.rand_float();
        count += 1;
        if count == 12 {
            xs.push(num - 6.0);
            count = 0;
            num = 0.0;
        }
    }
    xs
}

pub fn time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}
