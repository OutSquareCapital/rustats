use numpy::{ PyArray2, PyReadonlyArray2, IntoPyArray };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use std::collections::{ VecDeque };
use rayon::prelude::*;

#[pyfunction]
fn move_mean_old<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(|| {
        for (input_col, output_col) in input_columns.iter().zip(output_columns.iter_mut()) {
            let mut mean_sum: f32 = 0.0;
            let mut observation_count: usize = 0;

            for row in 0..length {
                let current: f32 = input_col[row];
                if !current.is_nan() {
                    observation_count += 1;
                    mean_sum += current;
                }

                if observation_count >= min_length {
                    output_col[row] = mean_sum / (observation_count as f32);
                }
            }

            for row in length..num_rows {
                let current: f32 = input_col[row];
                let precedent_idx: usize = row - length;
                let precedent: f32 = input_col[precedent_idx];

                if !current.is_nan() {
                    observation_count += 1;
                    mean_sum += current;
                }

                if !precedent.is_nan() {
                    observation_count -= 1;
                    mean_sum -= precedent;
                }

                if observation_count >= min_length {
                    output_col[row] = mean_sum / (observation_count as f32);
                }
            }
        }
    });

    Ok(output.into_pyarray(py).into())
}

#[pyfunction]
fn move_mean<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move ||{
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
            let mut mean_sum: f32 = 0.0;
            let mut observation_count: usize = 0;

            for row in 0..length {
                let current: f32 = input_col[row];
                if !current.is_nan() {
                    observation_count += 1;
                    mean_sum += current;
                }

                if observation_count >= min_length {
                    output_col[row] = mean_sum / (observation_count as f32);
                }
            }

            for row in length..num_rows {
                let current: f32 = input_col[row];
                let precedent_idx: usize = row - length;
                let precedent: f32 = input_col[precedent_idx];

                if !current.is_nan() {
                    observation_count += 1;
                    mean_sum += current;
                }

                if !precedent.is_nan() {
                    observation_count -= 1;
                    mean_sum -= precedent;
                }

                if observation_count >= min_length {
                    output_col[row] = mean_sum / (observation_count as f32);
                }
            }
        });
    });

    Ok(output.into_pyarray(py).into())
}

#[pyfunction]
fn move_max<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut max_deque = VecDeque::with_capacity(length);
                let mut observation_count: usize = 0;
                for row in 0..num_rows {
                    if row >= length {
                        let out_idx = row - length;
                        if !input_col[out_idx].is_nan() {
                            observation_count -= 1;
                            if let Some(&(_, pos)) = max_deque.front() {
                                if pos == out_idx {
                                    max_deque.pop_front();
                                }
                            }
                        }
                    }
                    let current = input_col[row];
                    if !current.is_nan() {
                        observation_count += 1;
                        while let Some(&(val, _)) = max_deque.back() {
                            if val < current {
                                max_deque.pop_back();
                            } else {
                                break;
                            }
                        }
                        max_deque.push_back((current, row));
                    }
                    if row >= length - 1 {
                        if observation_count >= min_length {
                            if let Some(&(val, _)) = max_deque.front() {
                                output_col[row] = val;
                            }
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pyfunction]
fn move_min<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut min_deque = VecDeque::with_capacity(length);
                let mut observation_count: usize = 0;
                for row in 0..num_rows {
                    if row >= length {
                        let out_idx = row - length;
                        if !input_col[out_idx].is_nan() {
                            observation_count -= 1;
                            if let Some(&(_, pos)) = min_deque.front() {
                                if pos == out_idx {
                                    min_deque.pop_front();
                                }
                            }
                        }
                    }
                    let current = input_col[row];
                    if !current.is_nan() {
                        observation_count += 1;
                        while let Some(&(val, _)) = min_deque.back() {
                            if val > current {
                                min_deque.pop_back();
                            } else {
                                break;
                            }
                        }
                        min_deque.push_back((current, row));
                    }
                    if row >= length - 1 {
                        if observation_count >= min_length {
                            if let Some(&(val, _)) = min_deque.front() {
                                output_col[row] = val;
                            }
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

struct IndexedHeap {
    heap: Vec<(f32, usize)>,
    positions: Vec<Option<usize>>,
    is_max_heap: bool,
}

impl IndexedHeap {
    fn new(capacity: usize, max_idx: usize, is_max_heap: bool) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            positions: vec![None; max_idx],
            is_max_heap,
        }
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    #[inline(always)]
    fn compare(&self, a: f32, b: f32) -> bool {
        let result: bool = a > b;
        result == self.is_max_heap
    }

    fn peek(&self) -> Option<(f32, usize)> {
        self.heap.first().copied()
    }

    fn push(&mut self, value: f32, idx: usize) {
        let pos: usize = self.heap.len();
        self.heap.push((value, idx));
        self.positions[idx] = Some(pos);
        self.sift_up(pos);
    }

    fn pop(&mut self) -> Option<(f32, usize)> {
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

    fn remove(&mut self, idx: usize) -> bool {
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

#[pyfunction]
fn move_median<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut small_heap: IndexedHeap = IndexedHeap::new(length, num_rows, true);
                let mut large_heap: IndexedHeap = IndexedHeap::new(length, num_rows, false);
                let mut window_q: VecDeque<(f32, usize)> = VecDeque::with_capacity(length + 1);
                let mut valid_count: usize = 0;

                for row in 0..num_rows {
                    let current_val: f32 = input_col[row];

                    window_q.push_back((current_val, row));

                    if !current_val.is_nan() {
                        valid_count += 1;

                        if let Some((max_small, _)) = small_heap.peek() {
                            if current_val > max_small {
                                large_heap.push(current_val, row);
                            } else {
                                small_heap.push(current_val, row);
                            }
                        } else {
                            small_heap.push(current_val, row);
                        }
                    }

                    if window_q.len() > length {
                        let (old_val, old_idx) = window_q.pop_front().unwrap();

                        if !old_val.is_nan() {
                            valid_count -= 1;

                            if small_heap.remove(old_idx) {
                            } else {
                                large_heap.remove(old_idx);
                            }
                        }
                    }
                    while small_heap.len() > large_heap.len() + 1 {
                        if let Some((val, idx)) = small_heap.pop() {
                            large_heap.push(val, idx);
                        }
                    }

                    while large_heap.len() > small_heap.len() {
                        if let Some((val, idx)) = large_heap.pop() {
                            small_heap.push(val, idx);
                        }
                    }
                    if window_q.len() >= length && valid_count >= min_length {
                        if small_heap.len() > large_heap.len() {
                            if let Some((val, _)) = small_heap.peek() {
                                output_col[row] = val;
                            }
                        } else if !small_heap.is_empty() {
                            let s_val: f32 = small_heap.peek().unwrap().0;
                            let l_val: f32 = large_heap.peek().unwrap().0;
                            output_col[row] = (s_val + l_val) / 2.0;
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(move_mean_old, module)?)?;
    module.add_function(wrap_pyfunction!(move_mean, module)?)?;
    module.add_function(wrap_pyfunction!(move_max, module)?)?;
    module.add_function(wrap_pyfunction!(move_min, module)?)?;
    module.add_function(wrap_pyfunction!(move_median, module)?)?;
    Ok(())
}
