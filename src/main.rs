use std::f32::consts::PI;

fn main() {
    let peak1 = Gauss {
        x0: 500.0,
        a: 4.0,
        s: 20.0,
    };
    let peak2 = Lorentz {
        x0: 200.0,
        a: 6.0,
        s: 50.0,
    };

    for (i, x) in (1..=1340)
        .map(|x| peak1.apply(x as f32, 0.0).then(&peak2))
        .enumerate()
    {
        println!("{},{}", i + 1, x.1)
    }
}

trait PeakFunction {
    fn apply(&self, x: f32, y0: f32) -> (f32, f32);
}

trait NextPeakFunction<T> {
    fn then(self, fun: &impl PeakFunction) -> Self;
}

impl NextPeakFunction<(f32, f32)> for (f32, f32) {
    fn then(self, fun: &impl PeakFunction) -> Self {
        fun.apply(self.0, self.1)
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
