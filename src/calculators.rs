use crate::accumulators as acc;
use crate::stats;
use numpy::ndarray as nd;
use std::collections::VecDeque;
use std::ops::{Add, AddAssign, Neg, Not, Sub, SubAssign};
pub struct WindowState {
    pub observations: usize,
    pub current: f64,
    pub precedent: f64,
    pub precedent_idx: usize,
}
impl WindowState {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            observations: 0,
            current: f64::NAN,
            precedent: f64::NAN,
            precedent_idx: 0,
        }
    }
    #[inline(always)]
    pub fn refresh(
        &mut self,
        input_col: &nd::ArrayBase<nd::ViewRepr<&f64>, nd::Dim<[usize; 1]>>,
        row: usize,
        length: usize,
    ) {
        self.current = input_col[row];
        self.precedent_idx = row.sub(length);
        self.precedent = input_col[self.precedent_idx];
    }
    #[inline(always)]
    pub fn compute_row<Calculator: StatCalculator>(&mut self, state: &mut Calculator::Accumulator) {
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
    pub fn compute_deque_row<Calculator: DequeStatCalculator>(
        &mut self,
        deque: &mut VecDeque<(f64, usize)>,
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

pub trait StatCalculator {
    type Accumulator;

    fn new() -> Self::Accumulator;
    fn add_value(state: &mut Self::Accumulator, value: f64);
    fn remove_value(state: &mut Self::Accumulator, value: f64);
    fn get(state: &Self::Accumulator, count: usize) -> f64;
}

pub trait DequeStatCalculator {
    fn new() -> VecDeque<(f64, usize)>;
    fn add_value(deque: &mut VecDeque<(f64, usize)>, value: f64, idx: usize);
}
pub struct Sum;
impl StatCalculator for Sum {
    type Accumulator = f64;

    fn new() -> Self::Accumulator {
        0.0
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        *state += value;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        *state -= value;
    }
    fn get(state: &Self::Accumulator, _count: usize) -> f64 {
        *state
    }
}

pub struct Mean;
impl StatCalculator for Mean {
    type Accumulator = f64;

    fn new() -> Self::Accumulator {
        0.0
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        *state += value;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        *state -= value;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        *state / (count as f64)
    }
}
pub struct Var;
impl StatCalculator for Var {
    type Accumulator = acc::Squared;

    fn new() -> Self::Accumulator {
        acc::Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::var(state.sum_simple, state.sum_square, count as f64)
    }
}

pub struct Stdev;
impl StatCalculator for Stdev {
    type Accumulator = acc::Squared;

    fn new() -> Self::Accumulator {
        acc::Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::stdev(state.sum_simple, state.sum_square, count as f64)
    }
}

pub struct Skewness;
impl StatCalculator for Skewness {
    type Accumulator = acc::Cubic;

    fn new() -> Self::Accumulator {
        acc::Cubic::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));

        let temp: f64 = value.powi(3).sub(state.compensation_cube);
        let total: f64 = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));

        let temp: f64 = value.neg().powi(3).sub(state.compensation_cube);
        let total: f64 = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::skew(
            state.sum_simple,
            state.sum_square,
            state.sum_cube,
            count as f64,
        )
    }
}
pub struct Kurtosis;
impl StatCalculator for Kurtosis {
    type Accumulator = acc::Quadratric;

    fn new() -> Self::Accumulator {
        acc::Quadratric::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.add_assign(value);
        state.sum_square.add_assign(value.powi(2));

        let temp: f64 = value.powi(3).sub(state.compensation_cube);
        let total: f64 = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;

        let temp: f64 = value.powi(4).sub(state.compensation_quad);
        let total: f64 = state.sum_quad.add(temp);
        state.compensation_quad = total.sub(state.sum_quad).sub(temp);
        state.sum_quad = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple.sub_assign(value);
        state.sum_square.sub_assign(value.powi(2));

        let temp: f64 = value.neg().powi(3).sub(state.compensation_cube);
        let total: f64 = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;

        let temp: f64 = value.neg().powi(4).sub(state.compensation_quad);
        let total: f64 = state.sum_quad.add(temp);
        state.compensation_quad = total.sub(state.sum_quad).sub(temp);
        state.sum_quad = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::kurtosis(
            state.sum_simple,
            state.sum_square,
            state.sum_cube,
            state.sum_quad,
            count as f64,
        )
    }
}

pub struct Min;
impl DequeStatCalculator for Min {
    fn new() -> VecDeque<(f64, usize)> {
        VecDeque::new()
    }

    fn add_value(deque: &mut VecDeque<(f64, usize)>, value: f64, idx: usize) {
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
impl DequeStatCalculator for Max {
    fn new() -> VecDeque<(f64, usize)> {
        VecDeque::new()
    }

    fn add_value(deque: &mut VecDeque<(f64, usize)>, value: f64, idx: usize) {
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

pub struct ValidCounter {
    greater_count: usize,
    equal_count: usize,
    pub valid_count: usize,
}
impl ValidCounter {
    pub fn new() -> Self {
        Self {
            greater_count: 0,
            equal_count: 1,
            valid_count: 1,
        }
    }

    pub fn add(&mut self, other: f64, current: f64) {
        if other.is_nan() {
            return;
        }
        self.valid_count.add_assign(1);
        if current.gt(&other) {
            self.greater_count.add_assign(1);
        } else if current.eq(&other) {
            self.equal_count.add_assign(1);
        }
    }

    pub fn reset(&mut self) {
        self.greater_count = 0;
        self.equal_count = 1;
        self.valid_count = 1;
    }

    pub fn get(&self) -> f64 {
        stats::rank(
            self.greater_count,
            self.equal_count,
            self.valid_count as f64,
        )
    }
}
