pub struct Indexed {
    heap: Vec<(f32, usize)>,
    positions: Vec<Option<usize>>,
    is_max_heap: bool,
}

impl Indexed {
    pub fn new(capacity: usize, max_idx: usize, is_max_heap: bool) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            positions: vec![None; max_idx],
            is_max_heap,
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    #[inline(always)]
    pub fn compare(&self, a: f32, b: f32) -> bool {
        let result: bool = a > b;
        result == self.is_max_heap
    }

    pub fn peek(&self) -> Option<(f32, usize)> {
        self.heap.first().copied()
    }

    pub fn push(&mut self, value: f32, idx: usize) {
        let pos: usize = self.heap.len();
        self.heap.push((value, idx));
        self.positions[idx] = Some(pos);
        self.sift_up(pos);
    }

    pub fn pop(&mut self) -> Option<(f32, usize)> {
        if self.heap.is_empty() {
            return None;
        }

        let result: (f32, usize) = self.heap[0];
        self.positions[result.1] = None;

        let last: (f32, usize) = self.heap.pop().unwrap();
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.positions[last.1] = Some(0);
            self.sift_down(0);
        }

        Some(result)
    }

    pub fn remove(&mut self, idx: usize) -> bool {
        if let Some(pos) = self.positions[idx] {
            self.positions[idx] = None;

            if pos == self.heap.len() - 1 {
                self.heap.pop();
            } else {
                let last: (f32, usize) = self.heap.pop().unwrap();
                self.heap[pos] = last;
                self.positions[last.1] = Some(pos);

                let parent: usize = pos.saturating_sub(1) / 2;
                if pos > 0 && self.compare(self.heap[pos].0, self.heap[parent].0) {
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

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent: usize = (pos - 1) / 2;
            if !self.compare(self.heap[pos].0, self.heap[parent].0) {
                break;
            }

            self.heap.swap(pos, parent);
            self.positions[self.heap[pos].1] = Some(pos);
            self.positions[self.heap[parent].1] = Some(parent);

            pos = parent;
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        let len: usize = self.heap.len();
        let node_value: f32 = self.heap[pos].0;
        let node_idx: usize = self.heap[pos].1;

        loop {
            let left: usize = 2 * pos + 1;
            if left >= len {
                break;
            }

            let right: usize = left + 1;
            let target = if right < len && self.compare(self.heap[right].0, self.heap[left].0) {
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
