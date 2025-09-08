use crate::accumulators as acc;
use crate::stats;
use numpy::ndarray as nd;
use std::collections::VecDeque;
use std::ops::{AddAssign, Not, Sub, SubAssign};
pub struct WindowState<T: nd::NdFloat> {
    pub observations: usize,
    pub current: T,
    pub precedent: T,
    pub precedent_idx: usize,
}
impl<T: nd::NdFloat> WindowState<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            observations: 0,
            current: T::nan(),
            precedent: T::nan(),
            precedent_idx: 0,
        }
    }
    #[inline(always)]
    pub fn refresh(
        &mut self,
        input_col: &nd::ArrayBase<nd::ViewRepr<&T>, nd::Dim<[usize; 1]>>,
        row: usize,
        length: usize,
    ) {
        self.current = input_col[row];
        self.precedent_idx = row.sub(length);
        self.precedent = input_col[self.precedent_idx];
    }
    #[inline(always)]
    pub fn compute_row<Calculator: StatCalculator<T>>(
        &mut self,
        state: &mut Calculator::Accumulator,
    ) {
        if self.current.is_nan().not() {
            self.observations.add_assign(1);
            Calculator::add_value(state, self.current);
        }

        if self.precedent.is_nan().not() {
            self.observations.sub_assign(1);
            Calculator::remove_value(state, self.precedent);
        }
    }
    #[inline(always)]
    pub fn compute_deque_row<Calculator: DequeStatCalculator<T>>(
        &mut self,
        deque: &mut VecDeque<(T, usize)>,
        row: usize,
    ) {
        if self.precedent.is_nan().not() {
            self.observations.sub_assign(1);
            if let Some(&(_, front_idx)) = deque.front() {
                if front_idx.eq(&self.precedent_idx) {
                    deque.pop_front();
                }
            }
        }

        if self.current.is_nan().not() {
            self.observations.add_assign(1);
            Calculator::add_value(deque, self.current, row);
        }
    }
}

pub trait StatCalculator<T: nd::NdFloat> {
    type Accumulator;

    fn new() -> Self::Accumulator;
    fn add_value(state: &mut Self::Accumulator, value: T);
    fn remove_value(state: &mut Self::Accumulator, value: T);
    fn get(state: &Self::Accumulator, count: usize) -> T;
}

pub trait DequeStatCalculator<T: nd::NdFloat> {
    fn new() -> VecDeque<(T, usize)>;
    fn add_value(deque: &mut VecDeque<(T, usize)>, value: T, idx: usize);
}
pub struct Sum;
impl<T: nd::NdFloat> StatCalculator<T> for Sum {
    type Accumulator = T;

    fn new() -> Self::Accumulator {
        T::zero()
    }
    fn add_value(state: &mut Self::Accumulator, value: T) {
        *state += value;
    }
    fn remove_value(state: &mut Self::Accumulator, value: T) {
        *state -= value;
    }
    fn get(state: &Self::Accumulator, _count: usize) -> T {
        *state
    }
}

pub struct Mean;
impl<T: nd::NdFloat> StatCalculator<T> for Mean {
    type Accumulator = T;

    fn new() -> Self::Accumulator {
        T::zero()
    }
    fn add_value(state: &mut Self::Accumulator, value: T) {
        *state += value;
    }
    fn remove_value(state: &mut Self::Accumulator, value: T) {
        *state -= value;
    }
    fn get(state: &Self::Accumulator, count: usize) -> T {
        *state / (T::from(count).unwrap())
    }
}
pub struct Var;
impl<T: nd::NdFloat> StatCalculator<T> for Var {
    type Accumulator = acc::Squared<T>;

    fn new() -> Self::Accumulator {
        acc::Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));
    }
    fn remove_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));
    }
    fn get(state: &Self::Accumulator, count: usize) -> T {
        stats::var(state.sum_simple, state.sum_square, T::from(count).unwrap())
    }
}

pub struct Stdev;
impl<T: nd::NdFloat> StatCalculator<T> for Stdev {
    type Accumulator = acc::Squared<T>;

    fn new() -> Self::Accumulator {
        acc::Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));
    }
    fn remove_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));
    }
    fn get(state: &Self::Accumulator, count: usize) -> T {
        stats::stdev(state.sum_simple, state.sum_square, T::from(count).unwrap())
    }
}

pub struct Skewness;
impl<T: nd::NdFloat> StatCalculator<T> for Skewness {
    type Accumulator = acc::Cubic<T>;

    fn new() -> Self::Accumulator {
        acc::Cubic::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));

        let temp = value.powi(3).sub(state.compensation_cube);
        let total = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));

        let temp = value.neg().powi(3).sub(state.compensation_cube);
        let total = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> T {
        stats::skew(
            state.sum_simple,
            state.sum_square,
            state.sum_cube,
            T::from(count).unwrap(),
        )
    }
}
pub struct Kurtosis;
impl<T: nd::NdFloat> StatCalculator<T> for Kurtosis {
    type Accumulator = acc::Quadratric<T>;

    fn new() -> Self::Accumulator {
        acc::Quadratric::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));

        let temp = value.powi(3).sub(state.compensation_cube);
        let total = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;

        let temp = value.powi(4).sub(state.compensation_quad);
        let total = state.sum_quad.add(temp);
        state.compensation_quad = total.sub(state.sum_quad).sub(temp);
        state.sum_quad = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: T) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));

        let temp = value.neg().powi(3).sub(state.compensation_cube);
        let total = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;

        let temp = value.neg().powi(4).sub(state.compensation_quad);
        let total = state.sum_quad.add(temp);
        state.compensation_quad = total.sub(state.sum_quad).sub(temp);
        state.sum_quad = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> T {
        stats::kurtosis(
            state.sum_simple,
            state.sum_square,
            state.sum_cube,
            state.sum_quad,
            T::from(count).unwrap(),
        )
    }
}

pub struct Min;
impl<T: nd::NdFloat> DequeStatCalculator<T> for Min {
    fn new() -> VecDeque<(T, usize)> {
        VecDeque::new()
    }

    fn add_value(deque: &mut VecDeque<(T, usize)>, value: T, idx: usize) {
        while let Some(&(val, _)) = deque.back() {
            if val.gt(&value) {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back((value, idx));
    }
}

pub struct Max;
impl<T: nd::NdFloat> DequeStatCalculator<T> for Max {
    fn new() -> VecDeque<(T, usize)> {
        VecDeque::new()
    }

    fn add_value(deque: &mut VecDeque<(T, usize)>, value: T, idx: usize) {
        while let Some(&(val, _)) = deque.back() {
            if val.gt(&value) {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back((value, idx));
    }
}
pub struct ValidCounter<T: nd::NdFloat> {
    greater_count: T,
    equal_count: T,
    pub valid_count: T,
}
impl<T: nd::NdFloat> ValidCounter<T> {
    pub fn new() -> Self {
        Self {
            greater_count: T::zero(),
            equal_count: T::one(),
            valid_count: T::one(),
        }
    }

    pub fn add(&mut self, other: T, current: T) {
        if other.is_nan() {
            return;
        }
        self.valid_count.add_assign(T::one());
        if current.gt(&other) {
            self.greater_count.add_assign(T::one());
        } else if current.eq(&other) {
            self.equal_count.add_assign(T::one());
        }
    }

    pub fn reset(&mut self) {
        self.greater_count = T::zero();
        self.equal_count = T::one();
        self.valid_count = T::one();
    }

    pub fn get(&self) -> T {
        stats::rank(self.greater_count, self.equal_count, self.valid_count)
    }
}
