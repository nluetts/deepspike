use std::fs::File;
use std::io::Read;

pub fn uniform(n: usize) -> std::io::Result<Vec<f32>> {
    let mut buf32 = [0u8; 4];
    let mut buf = vec![0; 4 * n];
    let mut xs = Vec::with_capacity(n);
    File::open("/dev/random")?.read_exact(&mut buf)?;
    for c in buf.chunks_exact(4) {
        for i in 0..4 {
            buf32[i] = c[i]
        }
        let x = u32::from_ne_bytes(buf32);
        xs.push(x as f32 / u32::MAX as f32)
    }
    Ok(xs)
}
