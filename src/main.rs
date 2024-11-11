/// We start by implementing a simple framework to simulate spectra based on
/// peak functions. The peak functions can be chained by putting them into
/// a vector and folding it while applying the function.
mod noise;

use std::f32::consts::PI;
use std::io::{BufWriter, Write};

use noise::time_ns;
use oorandom::Rand32;

fn main() {
    let mut rng = Rand32::new(1);
    let xs: Vec<f32> = (0..1340).map(|x| x as f32).collect();

    for n_spec in 1..=100 {
        let path = format!("train/train_spec_{}", n_spec);
        eprintln!("INFO: writing file {path}");
        let mut file = BufWriter::new(
            std::fs::File::create(&path)
                .unwrap_or_else(|err| panic!("Cannot write to file {path}: {err}")),
        );
        let peaks = random_peaks(20, time_ns());

        for _n_frames in 0..rng.rand_range(3..12) {
            let noise = noise::pseudo_normal(1340, None);
            let (xspec, yspec) = peaks.apply(xs.to_owned(), noise);
            for (x, y) in xspec.iter().zip(yspec.iter()) {
                let _ = writeln!(file, "{},{}", x, y)
                    .map_err(|err| eprintln!("WARN: could not write to file {path}: {err}"));
            }
        }
    }
}

trait PeakFunction {
    /// `x` value: for which x to calculate peak function  
    /// `y0` value: offset
    ///
    /// returns `(x, f(x) + y0)`
    fn apply(&self, xs: Vec<f32>, ys0: Vec<f32>) -> (Vec<f32>, Vec<f32>);
    fn center(&self) -> f32;
}

trait PeakPipeline {
    fn apply(&self, xs: Vec<f32>, ys0: Vec<f32>) -> (Vec<f32>, Vec<f32>);
}

impl PeakPipeline for Vec<Box<dyn PeakFunction>> {
    fn apply(&self, xs: Vec<f32>, ys0: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        self.iter().fold((xs, ys0), |(x, y), peak| peak.apply(x, y))
    }
}

struct Gauss {
    x0: f32,
    a: f32,
    s: f32,
}

impl Gauss {
    fn new(x0: f32, a: f32, s: f32) -> Self {
        Self { x0, a, s }
    }
}

impl PeakFunction for Gauss {
    fn apply(&self, xs: Vec<f32>, ys0: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let ys = xs
            .iter()
            .zip(ys0.iter())
            .map(|(xi, y0i)| {
                let norm = ((2.0 * PI * self.s.powi(2)).sqrt()).powi(-1);
                let exponent = -0.5 * ((xi - self.x0) / self.s).powi(2);
                y0i + self.a * norm * exponent.exp()
            })
            .collect();
        (xs, ys)
    }
    fn center(&self) -> f32 {
        self.x0
    }
}

struct Lorentz {
    x0: f32,
    a: f32,
    s: f32,
}

impl Lorentz {
    fn new(x0: f32, a: f32, s: f32) -> Self {
        Self { x0, a, s }
    }
}

impl PeakFunction for Lorentz {
    fn apply(&self, xs: Vec<f32>, ys0: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let ys = xs
            .iter()
            .zip(ys0.iter())
            .map(|(xi, y0i)| {
                y0i + self.a * (PI * self.s * (1.0 + ((xi - self.x0) / self.s).powi(2))).powi(-1)
            })
            .collect();
        (xs, ys)
    }
    fn center(&self) -> f32 {
        self.x0
    }
}

struct Skew<T: PeakFunction> {
    p: T,
    k: f32,
    skew_left: bool,
}

impl<T: PeakFunction> Skew<T> {
    fn left(p: T, k: f32) -> Self {
        Skew {
            p,
            k,
            skew_left: true,
        }
    }
    fn right(p: T, k: f32) -> Self {
        Skew {
            p,
            k,
            skew_left: false,
        }
    }
}

impl<T: PeakFunction> PeakFunction for Skew<T> {
    fn apply(&self, xs: Vec<f32>, ys0: Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let n = xs.len();
        let (xs, ys) = self.p.apply(xs, vec![0.0; n]);
        let ys = xs
            .iter()
            .zip(ys0.iter())
            .zip(ys.iter())
            .map(|((xi, y0i), yi)| {
                let mut sigmoid = 1.0 / (1.0 + (-self.k * (xi - self.p.center())).exp());
                if self.skew_left {
                    // invert the sigmoid
                    sigmoid = -sigmoid + 1.0
                }
                yi * sigmoid + y0i
            })
            .collect();
        (xs, ys)
    }
    fn center(&self) -> f32 {
        self.p.center()
    }
}

fn random_peaks(n: usize, seed: u64) -> Vec<Box<dyn PeakFunction>> {
    let mut peaks: Vec<Box<dyn PeakFunction>> = Vec::with_capacity(n);
    let mut rng = Rand32::new(seed);
    for _ in 0..n {
        let x0 = rng.rand_float() * 1340.0;
        let a = 100.0 + rng.rand_float() * 1000.0;
        let s = 1.0 + rng.rand_float() * 25.0;
        let k = rng.rand_float() / 10.0;
        let f = rng.rand_float();
        if f < 0.25 {
            peaks.push(Box::new(Skew::left(Gauss::new(x0, a, s), k)));
        } else if f < 0.5 {
            peaks.push(Box::new(Skew::right(Gauss::new(x0, a, s), k)));
        } else if f < 0.75 {
            peaks.push(Box::new(Skew::left(Lorentz::new(x0, a, s), k)));
        } else {
            peaks.push(Box::new(Skew::right(Lorentz::new(x0, a, s), k)));
        }
    }
    peaks
}
