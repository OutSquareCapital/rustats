use crate::stats;
use numpy::ndarray as nd;
use std::collections::VecDeque;
use std::ops::{Add, Div, Mul, Neg, Sub};
pub struct Squared {
    sum_simple: f64,
    sum_square: f64,
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
    sum_simple: f64,
    sum_square: f64,
    sum_cube: f64,
    compensation_cube: f64,
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
    sum_simple: f64,
    sum_square: f64,
    sum_cube: f64,
    compensation_cube: f64,
    sum_quad: f64,
    compensation_quad: f64,
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
        row: usize,
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
        state.sum_simple += value;
        state.sum_square += value.powi(2);
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple -= value;
        state.sum_square -= value.powi(2);
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::var(state.sum_simple, state.sum_square, count as f64)
    }
}

pub struct Stdev;
impl StatCalculator for Stdev {
    type Accumulator = Squared;

    fn new() -> Self::Accumulator {
        Squared::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple += value;
        state.sum_square += value.powi(2);
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple -= value;
        state.sum_square -= value.powi(2);
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::stdev(state.sum_simple, state.sum_square, count as f64)
    }
}

pub struct Skewness;
impl StatCalculator for Skewness {
    type Accumulator = Cubic;

    fn new() -> Self::Accumulator {
        Cubic::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple += value;
        state.sum_square += value.powi(2);

        let temp: f64 = value.powi(3).sub(state.compensation_cube);
        let total: f64 = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple -= value;
        state.sum_square -= value.powi(2);

        let temp: f64 = -value.neg().powi(3).sub(state.compensation_cube);
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
    type Accumulator = Quadratric;

    fn new() -> Self::Accumulator {
        Quadratric::new()
    }
    fn add_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple += value;
        state.sum_square += value.powi(2);

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
        state.sum_simple -= value;
        state.sum_square -= value.powi(2);

        let temp: f64 = -value.powi(3).sub(state.compensation_cube);
        let total: f64 = state.sum_cube.add(temp);
        state.compensation_cube = total.sub(state.sum_cube).sub(temp);
        state.sum_cube = total;

        let temp: f64 = -value.powi(4).sub(state.compensation_quad);
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
            if val < value {
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
        self.valid_count += 1;
        if current.gt(&other) {
            self.greater_count += 1;
        } else if current.eq(&other) {
            self.equal_count += 1;
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

pub struct HeapIndexed {
    pub heap: Vec<(f64, usize)>,
    positions: Vec<Option<usize>>,
    is_max_heap: bool,
}

impl HeapIndexed {
    pub fn new(capacity: usize, max_idx: usize, is_max_heap: bool) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            positions: vec![None; max_idx],
            is_max_heap,
        }
    }

    pub fn last(&self) -> f64 {
        self.peek().unwrap().0
    }

    #[inline(always)]
    pub fn compare(&self, a: f64, b: f64) -> bool {
        a.gt(&b).eq(&self.is_max_heap)
    }
    #[inline(always)]
    pub fn peek(&self) -> Option<(f64, usize)> {
        self.heap.first().copied()
    }

    pub fn push(&mut self, value: f64, idx: usize) {
        let pos: usize = self.heap.len();
        self.heap.push((value, idx));
        self.positions[idx] = Some(pos);
        self.sift_up(pos);
    }
    #[inline(always)]
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() {
            return None;
        }

        let result: (f64, usize) = self.heap[0];
        self.positions[result.1] = None;

        let last: (f64, usize) = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.positions[last.1] = Some(0);
            self.sift_down(0);
        }

        Some(result)
    }
    #[inline(always)]
    pub fn remove(&mut self, idx: usize) -> bool {
        if let Some(pos) = self.positions[idx] {
            self.positions[idx] = None;

            if pos.eq(&self.heap.len().sub(1)) {
                self.heap.pop();
            } else {
                let last: (f64, usize) = self.heap.pop().unwrap();
                self.heap[pos] = last;
                self.positions[last.1] = Some(pos);

                if pos.gt(&0)
                    && self.compare(self.heap[pos].0, self.heap[pos.saturating_sub(1).div(2)].0)
                {
                    self.sift_up(pos);
                } else {
                    self.sift_down(pos);
                }
            }
            true
        } else {
            false
        }
    }
    #[inline(always)]
    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent: usize = (pos.sub(1)).div(2);
            if !self.compare(self.heap[pos].0, self.heap[parent].0) {
                break;
            }

            self.heap.swap(pos, parent);
            self.positions[self.heap[pos].1] = Some(pos);
            self.positions[self.heap[parent].1] = Some(parent);

            pos = parent;
        }
    }
    #[inline(always)]
    fn sift_down(&mut self, mut pos: usize) {
        let len: usize = self.heap.len();
        let node_value: f64 = self.heap[pos].0;
        let node_idx: usize = self.heap[pos].1;

        loop {
            let left: usize = 2.mul(pos).add(1);
            if left.ge(&len) {
                break;
            }

            let right: usize = left.add(1);
            let target: usize =
                if right.ge(&len) && self.compare(self.heap[right].0, self.heap[left].0) {
                    right
                } else {
                    left
                };

            if !self.compare(self.heap[target].0, node_value) {
                break;
            }
            self.heap[pos] = self.heap[target];
            self.positions[self.heap[pos].1] = Some(pos);

            pos = target;
        }
        self.heap[pos] = (node_value, node_idx);
        self.positions[node_idx] = Some(pos);
    }
}

pub struct IndexedProcessor {
    pub small_heap: HeapIndexed,
    pub large_heap: HeapIndexed,
    pub deque: VecDeque<(f64, usize)>,
}
impl IndexedProcessor {
    pub fn new(capacity: usize, max_idx: usize) -> Self {
        let small_heap = HeapIndexed::new(capacity, max_idx, true);
        let large_heap = HeapIndexed::new(capacity, max_idx, false);
        let deque = VecDeque::with_capacity(capacity.add(1));

        Self {
            small_heap,
            large_heap,
            deque,
        }
    }
    pub fn push_values(&mut self, current: f64, row: usize) {
        {
            if let Some((max_small, _)) = self.small_heap.peek() {
                if current.gt(&max_small) {
                    self.large_heap.push(current, row);
                } else {
                    self.small_heap.push(current, row);
                }
            } else {
                self.small_heap.push(current, row);
            }
        }
    }
    pub fn equilibrate(&mut self) {
        while self
            .small_heap
            .heap
            .len()
            .gt(&self.large_heap.heap.len().add(1))
        {
            if let Some((val, idx)) = self.small_heap.pop() {
                self.large_heap.push(val, idx);
            }
        }

        while self.large_heap.heap.len().gt(&self.small_heap.heap.len()) {
            if let Some((val, idx)) = self.large_heap.pop() {
                self.small_heap.push(val, idx);
            }
        }
    }

    pub fn remove(&mut self, window: &mut WindowState, length: usize) {
        if self.deque.len() > length {
            (window.precedent, window.precedent_idx) = self.deque.pop_front().unwrap();

            if !window.precedent.is_nan() {
                window.observations -= 1;

                if self.small_heap.remove(window.precedent_idx) {
                } else {
                    self.large_heap.remove(window.precedent_idx);
                }
            }
        }
    }
    pub fn check(&self) -> bool {
        self.small_heap.heap.len().gt(&self.large_heap.heap.len())
    }

    pub fn get(&self) -> f64 {
        (self.small_heap.last().add(self.large_heap.last())).div(2.0)
    }
}
