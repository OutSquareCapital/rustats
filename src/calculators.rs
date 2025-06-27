use crate::stats;
use std::collections::VecDeque;
use numpy::ndarray::{ ArrayBase, ViewRepr, Dim };
pub struct Squared {
    sum: f64,
    sum_squared: f64,
}

impl Squared {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            sum_squared: 0.0,
        }
    }
}

pub struct Cubic {
    sum: f64,
    sum_squared: f64,
    sum_cubic: f64,
    compensation_cubic: f64,
}

impl Cubic {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            sum_squared: 0.0,
            sum_cubic: 0.0,
            compensation_cubic: 0.0,
        }
    }
}

pub struct Quadratric {
    sum: f64,
    sum_squared: f64,
    sum_cubic: f64,
    compensation_cubic: f64,
    sum_quad: f64,
    compensation_quad: f64,
}

impl Quadratric {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            sum_squared: 0.0,
            sum_cubic: 0.0,
            compensation_cubic: 0.0,
            sum_quad: 0.0,
            compensation_quad: 0.0,
        }
    }
}

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
        input_col: &ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>>,
        row: usize,
        length: usize
    ) {
        self.current = input_col[row];
        self.precedent_idx = row - length;
        self.precedent = input_col[self.precedent_idx];
    }
    #[inline(always)]
    pub fn compute_row<Calculator: StatCalculator>(&mut self, state: &mut Calculator::Accumulator) {
        if !self.current.is_nan() {
            self.observations += 1;
            Calculator::add_value(state, self.current);
        }

        if !self.precedent.is_nan() {
            self.observations -= 1;
            Calculator::remove_value(state, self.precedent);
        }
    }
    #[inline(always)]
    pub fn compute_deque_row<Calculator: DequeStatCalculator>(
        &mut self,
        deque: &mut VecDeque<(f64, usize)>,
        row: usize
    ) {
        if !self.precedent.is_nan() {
            self.observations -= 1;
            if let Some(&(_, front_idx)) = deque.front() {
                if front_idx == self.precedent_idx {
                    deque.pop_front();
                }
            }
        }

        if !self.current.is_nan() {
            self.observations += 1;
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
    type Accumulator = Squared;

    fn new() -> Self::Accumulator {
        Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum += value;
        state.sum_squared += value.powi(2);
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum -= value;
        state.sum_squared -= value.powi(2);
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::var(state.sum, state.sum_squared, count as f64)
    }
}

pub struct Stdev;
impl StatCalculator for Stdev {
    type Accumulator = Squared;

    fn new() -> Self::Accumulator {
        Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum += value;
        state.sum_squared += value.powi(2);
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum -= value;
        state.sum_squared -= value.powi(2);
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::stdev(state.sum, state.sum_squared, count as f64)
    }
}

pub struct Skewness;
impl StatCalculator for Skewness {
    type Accumulator = Cubic;

    fn new() -> Self::Accumulator {
        Cubic::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum += value;
        state.sum_squared += value.powi(2);

        let temp: f64 = value.powi(3) - state.compensation_cubic;
        let total: f64 = state.sum_cubic + temp;
        state.compensation_cubic = total - state.sum_cubic - temp;
        state.sum_cubic = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum -= value;
        state.sum_squared -= value.powi(2);

        let temp: f64 = -value.powi(3) - state.compensation_cubic;
        let total: f64 = state.sum_cubic + temp;
        state.compensation_cubic = total - state.sum_cubic - temp;
        state.sum_cubic = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::skew(state.sum, state.sum_squared, state.sum_cubic, count as f64)
    }
}
pub struct Kurtosis;
impl StatCalculator for Kurtosis {
    type Accumulator = Quadratric;

    fn new() -> Self::Accumulator {
        Quadratric::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum += value;
        state.sum_squared += value.powi(2);

        let temp: f64 = value.powi(3) - state.sum_cubic;
        let total: f64 = state.sum_cubic + temp;
        state.compensation_cubic = total - state.sum_cubic - temp;
        state.sum_cubic = total;

        let temp: f64 = value.powi(4) - state.sum_quad;
        let total: f64 = state.compensation_quad + temp;
        state.compensation_quad = total - state.sum_quad - temp;
        state.sum_quad = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum -= value;
        state.sum_squared -= value.powi(2);

        let temp: f64 = -value.powi(3) - state.compensation_cubic;
        let total: f64 = state.sum_cubic + temp;
        state.compensation_cubic = total - state.sum_cubic - temp;
        state.sum_cubic = total;

        let temp: f64 = -value.powi(4) - state.sum_quad;
        let total: f64 = state.compensation_quad + temp;
        state.compensation_quad = total - state.compensation_quad - temp;
        state.sum_quad = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::kurtosis(state.sum, state.sum_squared, state.sum_cubic, state.sum_quad, count as f64)
    }
}

pub struct Min;
impl DequeStatCalculator for Min {
    fn new() -> VecDeque<(f64, usize)> {
        VecDeque::new()
    }

    fn add_value(deque: &mut VecDeque<(f64, usize)>, value: f64, idx: usize) {
        while let Some(&(val, _)) = deque.back() {
            if val > value {
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
            if val < value {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back((value, idx));
    }
}

pub struct Rank {
    greater_count: usize,
    equal_count: usize,
    pub valid_count: usize,
}
impl Rank {
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
        self.valid_count += 1;
        if current > other {
            self.greater_count += 2;
        } else if current == other {
            self.equal_count += 1;
        }
    }

    pub fn get(&self) -> f64 {
        stats::rank(self.greater_count, self.equal_count, self.valid_count as f64)
    }
}
