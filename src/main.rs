/// We start by implementing a simple framework to simulate spectra based on
/// peak functions. The peak functions can be chained by putting them into
/// a vector and folding it while applying the function.
use std::f32::consts::PI;

fn main() {
    let peak1 = Box::new(Gauss {
        x0: 500.0,
        a: 4.0,
        s: 20.0,
    });
    let peak2 = Box::new(Lorentz {
        x0: 200.0,
        a: 6.0,
        s: 50.0,
    });
    let peaks: Vec<Box<dyn PeakFunction>> = vec![peak1, peak2];

    for (i, x) in (1..=1340).map(|x| peaks.apply(x as f32, 0.0)).enumerate() {
        println!("{},{}", i + 1, x.1)
    }
}

trait PeakFunction {
    /// `x` value: for which x to calculate peak function  
    /// `y0` value: offset
    ///
    /// returns `(x, f(x) + y0)`
    fn apply(&self, x: f32, y0: f32) -> (f32, f32);
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

impl PeakFunction for Gauss {
    fn apply(&self, x: f32, y0: f32) -> (f32, f32) {
        let norm = ((2.0 * PI * self.s.powi(2)).sqrt()).powi(-1);
        let exponent = -0.5 * ((x - self.x0) / self.s).powi(2);
        (x, y0 + self.a * norm * exponent.exp())
    }
}

struct Lorentz {
    x0: f32,
    a: f32,
    s: f32,
}

impl PeakFunction for Lorentz {
    fn apply(&self, x: f32, y0: f32) -> (f32, f32) {
        (
            x,
            y0 + self.a * (PI * self.s * (1.0 + ((x - self.x0) / self.s).powi(2))).powi(-1),
        )
    }
}
