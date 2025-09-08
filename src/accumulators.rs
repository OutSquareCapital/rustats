use ndarray::NdFloat;
pub struct Squared<T: NdFloat> {
    pub sum_simple: T,
    pub sum_square: T,
}

impl<T: NdFloat> Squared<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum_simple: T::zero(),
            sum_square: T::zero(),
        }
    }
}

pub struct Cubic<T: NdFloat> {
    pub sum_simple: T,
    pub sum_square: T,
    pub sum_cube: T,
    pub compensation_cube: T,
}

impl<T: NdFloat> Cubic<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum_simple: T::zero(),
            sum_square: T::zero(),
            sum_cube: T::zero(),
            compensation_cube: T::zero(),
        }
    }
}

pub struct Quadratric<T: NdFloat> {
    pub sum_simple: T,
    pub sum_square: T,
    pub sum_cube: T,
    pub compensation_cube: T,
    pub sum_quad: T,
    pub compensation_quad: T,
}

impl<T: NdFloat> Quadratric<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum_simple: T::zero(),
            sum_square: T::zero(),
            sum_cube: T::zero(),
            compensation_cube: T::zero(),
            sum_quad: T::zero(),
            compensation_quad: T::zero(),
        }
    }
}
