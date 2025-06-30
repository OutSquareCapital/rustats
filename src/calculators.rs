use crate::stats;
use std::collections::VecDeque;
use numpy::ndarray as nd;
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

        let temp: f64 = value.powi(3) - state.compensation_cube;
        let total: f64 = state.sum_cube + temp;
        state.compensation_cube = total - state.sum_cube - temp;
        state.sum_cube = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple -= value;
        state.sum_square -= value.powi(2);

        let temp: f64 = -value.powi(3) - state.compensation_cube;
        let total: f64 = state.sum_cube + temp;
        state.compensation_cube = total - state.sum_cube - temp;
        state.sum_cube = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::skew(state.sum_simple, state.sum_square, state.sum_cube, count as f64)
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

        let temp: f64 = value.powi(3) - state.compensation_cube;
        let total: f64 = state.sum_cube + temp;
        state.compensation_cube = total - state.sum_cube - temp;
        state.sum_cube = total;

        let temp: f64 = value.powi(4) - state.compensation_quad;
        let total: f64 = state.sum_quad + temp;
        state.compensation_quad = total - state.sum_quad - temp;
        state.sum_quad = total;
    }
    fn remove_value(state: &mut Self::Accumulator, value: f64) {
        state.sum_simple -= value;
        state.sum_square -= value.powi(2);

        let temp: f64 = -value.powi(3) - state.compensation_cube;
        let total: f64 = state.sum_cube + temp;
        state.compensation_cube = total - state.sum_cube - temp;
        state.sum_cube = total;

        let temp: f64 = -value.powi(4) - state.compensation_quad;
        let total: f64 = state.sum_quad + temp;
        state.compensation_quad = total - state.sum_quad - temp;
        state.sum_quad = total;
    }
    fn get(state: &Self::Accumulator, count: usize) -> f64 {
        stats::kurtosis(
            state.sum_simple,
            state.sum_square,
            state.sum_cube,
            state.sum_quad,
            count as f64
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

pub struct DancingLinks {
    values: Vec<f64>,
    pi: Vec<u32>,
    prev: Vec<u32>,
    next: Vec<u32>,
    current_position: usize,
    current_index: usize,
    n_element: usize,
    tail: usize,
    index_map: Vec<usize>,
}

impl DancingLinks {
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            pi: Vec::with_capacity(capacity),
            prev: vec![0; capacity + 1],
            next: vec![0; capacity + 1],
            current_position: 0,
            current_index: 0,
            n_element: 0,
            tail: capacity,
            index_map: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn init_links(&mut self) {
        let mut p = self.tail;

        for &q in self.pi.iter() {
            self.next[p] = q;
            self.prev[q as usize] = p as u32;
            p = q as usize;
        }

        self.next[p] = self.tail as u32;
        self.prev[self.tail] = p as u32;
    }

    #[inline(always)]
    pub fn delete_link(&mut self, i: usize) {
        self.next[self.prev[i] as usize] = self.next[i];
        self.prev[self.next[i] as usize] = self.prev[i];
        self.n_element -= 1;
    }
    #[inline(always)]
    pub fn traverse_to_index(&mut self, i: usize) {
        match (i as i64) - (self.current_index as i64) {
            0 => {}
            -1 => {
                self.current_index -= 1;
                self.current_position = self.prev[self.current_position] as usize;
            }
            1 => self.advance(),
            i64::MIN..=0 => {
                for _ in i..self.current_index {
                    self.current_position = self.prev[self.current_position] as usize;
                }
                self.current_index = i;
            }
            _ => {
                for _ in self.current_index..i {
                    self.current_position = self.next[self.current_position] as usize;
                }
                self.current_index = i;
            }
        }
    }

    #[inline(always)]
    pub fn advance(&mut self) {
        if self.current_index < self.n_element {
            self.current_index += 1;
            self.current_position = self.next[self.current_position] as usize;
        }
    }

    #[inline(always)]
    pub fn at_end(&self) -> bool {
        self.current_position == self.tail
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n_element == 0
    }

    #[inline(always)]
    pub fn peek(&self) -> Option<f64> {
        if self.at_end() { None } else { Some(self.values[self.current_position]) }
    }
    pub fn fill(&mut self, values: &[f64]) {
        self.clear();

        for (i, &v) in values.iter().enumerate() {
            if !v.is_nan() {
                self.values.push(v);
                self.index_map.push(i);
            }
        }

        let n = self.values.len();
        self.pi = (0..n as u32).collect();
        self.pi.sort_by(|&a, &b| {
            self.values[a as usize].partial_cmp(&self.values[b as usize]).unwrap()
        });

        self.prev.resize(n + 1, 0);
        self.next.resize(n + 1, 0);
        self.n_element = n;
        self.tail = n;

        self.init_links();

        if !self.is_empty() {
            self.current_index = 0;
            self.current_position = self.next[self.tail] as usize;
            let mid = self.n_element / 2;
            self.traverse_to_index(mid);
        }
    }

    pub fn remove_by_original_index(&mut self, original_idx: usize) -> bool {
        let pos = self.index_map.iter().position(|&idx| idx == original_idx);

        if let Some(pos) = pos {
            self.delete_link(pos);
            if self.current_index > 0 && pos < self.current_index {
                self.current_index -= 1;
            }
            return true;
        }

        false
    }

    pub fn add_with_index(&mut self, value: f64, original_idx: usize) {
        let pos = self.values.len();
        self.values.push(value);
        self.index_map.push(original_idx);

        let insert_pos = match
            self.pi.binary_search_by(|&i| { self.values[i as usize].partial_cmp(&value).unwrap() })
        {
            Ok(pos) => pos,
            Err(pos) => pos,
        };

        self.pi.insert(insert_pos, pos as u32);

        if self.n_element == 0 {
            self.next[self.tail] = pos as u32;
            self.prev[pos] = self.tail as u32;
            self.next[pos] = self.tail as u32;
            self.prev[self.tail] = pos as u32;
            self.current_position = pos;
            self.current_index = 0;
        } else {
            if insert_pos > 0 {
                let prev_idx = self.pi[insert_pos - 1] as usize;
                let next_idx = self.next[prev_idx];

                self.next[prev_idx] = pos as u32;
                self.prev[pos] = prev_idx as u32;
                self.next[pos] = next_idx;
                self.prev[next_idx as usize] = pos as u32;
            } else {
                let next_idx = self.next[self.tail];

                self.next[self.tail] = pos as u32;
                self.prev[pos] = self.tail as u32;
                self.next[pos] = next_idx;
                self.prev[next_idx as usize] = pos as u32;
            }

            let new_mid = self.n_element / 2;
            if new_mid != self.current_index {
                self.traverse_to_index(new_mid);
            }
        }

        self.n_element += 1;
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.pi.clear();
        self.prev.resize(1, 0);
        self.next.resize(1, 0);
        self.current_position = 0;
        self.current_index = 0;
        self.n_element = 0;
        self.index_map.clear();
    }

    pub fn median(&mut self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }

        let mid = self.n_element / 2;
        self.traverse_to_index(mid);

        if self.n_element % 2 == 0 && self.n_element > 1 {
            let v1 = self.peek().unwrap();
            self.traverse_to_index(mid - 1);
            let v2 = self.peek().unwrap();
            (v1 + v2) / 2.0
        } else {
            self.peek().unwrap()
        }
    }

    pub fn len(&self) -> usize {
        self.n_element
    }
}
