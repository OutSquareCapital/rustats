use crate::stats;
use std::collections::VecDeque;

pub trait StatCalculator {
    type State;

    fn new() -> Self::State;
    fn add_value(state: &mut Self::State, value: f64);
    fn remove_value(state: &mut Self::State, value: f64);
    fn get(state: &Self::State, count: usize) -> f32;
}

pub trait DequeStatCalculator {
    fn new() -> VecDeque<(f32, usize)>;
    fn add_with_index(deque: &mut VecDeque<(f32, usize)>, value: f32, idx: usize);
}
pub struct Sum;
impl StatCalculator for Sum {
    type State = f64;

    fn new() -> Self::State {
        0.0
    }
    fn add_value(state: &mut Self::State, value: f64) {
        *state += value;
    }
    fn remove_value(state: &mut Self::State, value: f64) {
        *state -= value;
    }
    fn get(state: &Self::State, _count: usize) -> f32 {
        *state as f32
    }
}

pub struct Mean;
impl StatCalculator for Mean {
    type State = f64;

    fn new() -> Self::State {
        0.0
    }
    fn add_value(state: &mut Self::State, value: f64) {
        *state += value;
    }
    fn remove_value(state: &mut Self::State, value: f64) {
        *state -= value;
    }
    fn get(state: &Self::State, count: usize) -> f32 {
        stats::mean(*state, count) as f32
    }
}
pub struct Var;
impl StatCalculator for Var {
    type State = (f64, f64);

    fn new() -> Self::State {
        (0.0, 0.0)
    }
    fn add_value(state: &mut Self::State, value: f64) {
        state.0 += value;
        state.1 += value * value;
    }
    fn remove_value(state: &mut Self::State, value: f64) {
        state.0 -= value;
        state.1 -= value * value;
    }
    fn get(state: &Self::State, count: usize) -> f32 {
        stats::var(state.0, state.1, count) as f32
    }
}

pub struct Stdev;
impl StatCalculator for Stdev {
    type State = (f64, f64);

    fn new() -> Self::State {
        (0.0, 0.0)
    }
    fn add_value(state: &mut Self::State, value: f64) {
        state.0 += value;
        state.1 += value * value;
    }
    fn remove_value(state: &mut Self::State, value: f64) {
        state.0 -= value;
        state.1 -= value * value;
    }
    fn get(state: &Self::State, count: usize) -> f32 {
        stats::stdev(state.0, state.1, count) as f32
    }
}

pub struct Skewness;
impl StatCalculator for Skewness {
    type State = (f64, f64, f64, f64);

    fn new() -> Self::State {
        (0.0, 0.0, 0.0, 0.0)
    }
    fn add_value(state: &mut Self::State, value: f64) {
        state.0 += value;
        state.1 += value.powi(2);

        let temp: f64 = value.powi(3) - state.3;
        let total: f64 = state.2 + temp;
        state.3 = total - state.2 - temp;
        state.2 = total;
    }
    fn remove_value(state: &mut Self::State, value: f64) {
        state.0 -= value;
        state.1 -= value.powi(2);

        let temp: f64 = -value.powi(3) - state.3;
        let total: f64 = state.2 + temp;
        state.3 = total - state.2 - temp;
        state.2 = total;
    }
    fn get(state: &Self::State, count: usize) -> f32 {
        stats::skew(state.0, state.1, state.2, count) as f32
    }
}
pub struct Kurtosis;
impl StatCalculator for Kurtosis {
    type State = (f64, f64, f64, f64, f64, f64);

    fn new() -> Self::State {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
    fn add_value(state: &mut Self::State, value: f64) {
        state.0 += value;
        state.1 += value.powi(2);

        let temp: f64 = value.powi(3) - state.3;
        let total: f64 = state.2 + temp;
        state.3 = total - state.2 - temp;
        state.2 = total;

        let temp: f64 = value.powi(4) - state.5;
        let total: f64 = state.4 + temp;
        state.5 = total - state.4 - temp;
        state.4 = total;
    }
    fn remove_value(state: &mut Self::State, value: f64) {
        state.0 -= value;
        state.1 -= value.powi(2);

        let temp: f64 = -value.powi(3) - state.3;
        let total: f64 = state.2 + temp;
        state.3 = total - state.2 - temp;
        state.2 = total;

        let temp: f64 = -value.powi(4) - state.5;
        let total: f64 = state.4 + temp;
        state.5 = total - state.4 - temp;
        state.4 = total;
    }
    fn get(state: &Self::State, count: usize) -> f32 {
        stats::kurtosis(state.0, state.1, state.2, state.4, count) as f32
    }
}

pub struct Min;
impl DequeStatCalculator for Min {
    fn new() -> VecDeque<(f32, usize)> {
        VecDeque::new()
    }

    fn add_with_index(deque: &mut VecDeque<(f32, usize)>, value: f32, idx: usize) {
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
    fn new() -> VecDeque<(f32, usize)> {
        VecDeque::new()
    }

    fn add_with_index(deque: &mut VecDeque<(f32, usize)>, value: f32, idx: usize) {
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
