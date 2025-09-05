pub struct Squared {
    pub sum_simple: f64,
    pub sum_square: f64,
}

impl Squared {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum_simple: 0.0,
            sum_square: 0.0,
        }
    }
}

pub struct Cubic {
    pub sum_simple: f64,
    pub sum_square: f64,
    pub sum_cube: f64,
    pub compensation_cube: f64,
}

impl Cubic {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum_simple: 0.0,
            sum_square: 0.0,
            sum_cube: 0.0,
            compensation_cube: 0.0,
        }
    }
}

pub struct Quadratric {
    pub sum_simple: f64,
    pub sum_square: f64,
    pub sum_cube: f64,
    pub compensation_cube: f64,
    pub sum_quad: f64,
    pub compensation_quad: f64,
}

impl Quadratric {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum_simple: 0.0,
            sum_square: 0.0,
            sum_cube: 0.0,
            compensation_cube: 0.0,
            sum_quad: 0.0,
            compensation_quad: 0.0,
        }
    }
}
