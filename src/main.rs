/// We start by implementing a simple framework to simulate spectra based on
/// peak functions. The peak functions can be chained by putting them into
/// a vector and folding it while applying the function.
mod noise;

use std::f32::consts::PI;

fn main() {
    let peaks: Vec<Box<dyn PeakFunction>> = vec![
        Box::new(Skew::left(Gauss::new(1000.0, 6.0, 110.0), 0.18)),
        Box::new(Skew::left(Gauss::new(1050.0, 6.5, 115.0), 0.2)),
        Box::new(Skew::left(Gauss::new(1100.0, 7.0, 120.0), 0.22)),
        Box::new(Skew::left(Gauss::new(1150.0, 7.5, 125.0), 0.24)),
        Box::new(Skew::left(Gauss::new(1200.0, 8.0, 130.0), 0.26)),
        Box::new(Skew::left(Gauss::new(1250.0, 8.5, 135.0), 0.28)),
        Box::new(Skew::left(Gauss::new(800.0, 4.0, 90.0), 0.1)),
        Box::new(Skew::left(Gauss::new(850.0, 4.5, 95.0), 0.12)),
        Box::new(Skew::left(Gauss::new(900.0, 5.0, 100.0), 0.14)),
        Box::new(Skew::left(Gauss::new(950.0, 5.5, 105.0), 0.16)),
        Box::new(Skew::right(Lorentz::new(200.0, 6.0, 50.0), 1e-2)),
        Box::new(Skew::right(Lorentz::new(220.0, 6.5, 55.0), 1.2e-2)),
        Box::new(Skew::right(Lorentz::new(240.0, 7.0, 60.0), 1.4e-2)),
        Box::new(Skew::right(Lorentz::new(260.0, 7.5, 65.0), 1.6e-2)),
        Box::new(Skew::right(Lorentz::new(280.0, 8.0, 70.0), 1.8e-2)),
        Box::new(Skew::right(Lorentz::new(300.0, 8.5, 75.0), 2e-2)),
        Box::new(Skew::right(Lorentz::new(320.0, 9.0, 80.0), 2.2e-2)),
        Box::new(Skew::right(Lorentz::new(340.0, 9.5, 85.0), 2.4e-2)),
        Box::new(Skew::right(Lorentz::new(360.0, 10.0, 90.0), 2.6e-2)),
        Box::new(Skew::right(Lorentz::new(380.0, 10.5, 95.0), 2.8e-2)),
    ];
    let noise = noise::uniform(1340).unwrap();

    for (i, x) in (0..1340)
        .zip(noise)
        .map(|(x, n)| peaks.apply(x as f32, n / 300.0))
        .enumerate()
    {
        println!("{},{}", i + 1, x.1)
    }
}

trait PeakFunction {
    /// `x` value: for which x to calculate peak function  
    /// `y0` value: offset
    ///
    /// returns `(x, f(x) + y0)`
    fn apply(&self, x: f32, y0: f32) -> (f32, f32);
    fn center(&self) -> f32;
}

trait PeakPipeline {
    fn apply(&self, x: f32, y0: f32) -> (f32, f32);
}

impl PeakPipeline for Vec<Box<dyn PeakFunction>> {
    fn apply(&self, x: f32, y0: f32) -> (f32, f32) {
        self.iter().fold((x, y0), |(x, y), peak| peak.apply(x, y))
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
    fn apply(&self, x: f32, y0: f32) -> (f32, f32) {
        let norm = ((2.0 * PI * self.s.powi(2)).sqrt()).powi(-1);
        let exponent = -0.5 * ((x - self.x0) / self.s).powi(2);
        // dbg!(sigmoid);
        (x, y0 + self.a * norm * exponent.exp())
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
    fn apply(&self, x: f32, y0: f32) -> (f32, f32) {
        (
            x,
            y0 + self.a * (PI * self.s * (1.0 + ((x - self.x0) / self.s).powi(2))).powi(-1),
        )
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
    fn apply(&self, x: f32, y0: f32) -> (f32, f32) {
        let (_, y) = self.p.apply(x, 0.0);
        let mut sigmoid = 1.0 / (1.0 + (-self.k * (x - self.p.center())).exp());
        if self.skew_left {
            // invert the sigmoid
            sigmoid = -sigmoid + 1.0
        }
        (x, y * sigmoid + y0)
    }
    fn center(&self) -> f32 {
        self.p.center()
    }
}
