use crate::calculators::WindowState;
use std::collections::VecDeque;
use std::ops::{Add, Div, Mul, Not, Sub, SubAssign};

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
        if self.heap.is_empty().not() {
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
        while pos.gt(&0) {
            let parent: usize = (pos.sub(1)).div(2);
            if self.compare(self.heap[pos].0, self.heap[parent].0).not() {
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

            if self.compare(self.heap[target].0, node_value).not() {
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
        if self.deque.len().gt(&length) {
            (window.precedent, window.precedent_idx) = self.deque.pop_front().unwrap();

            if window.precedent.is_nan().not() {
                window.observations.sub_assign(1);

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
