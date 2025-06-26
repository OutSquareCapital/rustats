use std::collections::VecDeque;
use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use rayon::prelude::*;
use std::cmp::Ordering;
use crate::calculators;

#[pyfunction]
pub fn move_median<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    if parallel {
        move_median_parallel(py, array, length, min_length)
    } else {
        move_median_single(py, array, length, min_length)
    }
}

struct Indexed {
    heap: Vec<(f64, usize)>,
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

    #[inline(always)]
    pub fn compare(&self, a: f64, b: f64) -> bool {
        let result: bool = a > b;
        result == self.is_max_heap
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

            if pos == self.heap.len() - 1 {
                self.heap.pop();
            } else {
                let last: (f64, usize) = self.heap.pop().unwrap();
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
    #[inline(always)]
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
    #[inline(always)]
    fn sift_down(&mut self, mut pos: usize) {
        let len: usize = self.heap.len();
        let node_value: f64 = self.heap[pos].0;
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

fn move_median_parallel<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut small_heap = Indexed::new(length, num_rows, true);
                let mut large_heap = Indexed::new(length, num_rows, false);
                let mut window_q: VecDeque<(f64, usize)> = VecDeque::with_capacity(length + 1);
                let mut window = calculators::WindowState::new();

                for row in 0..length {
                    window.current = input_col[row];

                    window_q.push_back((window.current, row));

                    if !window.current.is_nan() {
                        window.observations += 1;

                        if let Some((max_small, _)) = small_heap.peek() {
                            if window.current > max_small {
                                large_heap.push(window.current, row);
                            } else {
                                small_heap.push(window.current, row);
                            }
                        } else {
                            small_heap.push(window.current, row);
                        }
                    }

                    while small_heap.heap.len() > large_heap.heap.len() + 1 {
                        if let Some((val, idx)) = small_heap.pop() {
                            large_heap.push(val, idx);
                        }
                    }

                    while large_heap.heap.len() > small_heap.heap.len() {
                        if let Some((val, idx)) = large_heap.pop() {
                            small_heap.push(val, idx);
                        }
                    }
                    if window.observations >= min_length {
                        if small_heap.heap.len() > large_heap.heap.len() {
                            if let Some((val, _)) = small_heap.peek() {
                                output_col[row] = val;
                            }
                        } else if !small_heap.heap.is_empty() {
                            let s_val: f64 = small_heap.peek().unwrap().0;
                            let l_val: f64 = large_heap.peek().unwrap().0;
                            output_col[row] = (s_val + l_val) / 2.0;
                        }
                    }
                }

                for row in length..num_rows {
                    window.current = input_col[row];

                    window_q.push_back((window.current, row));

                    if !window.current.is_nan() {
                        window.observations += 1;

                        if let Some((max_small, _)) = small_heap.peek() {
                            if window.current > max_small {
                                large_heap.push(window.current, row);
                            } else {
                                small_heap.push(window.current, row);
                            }
                        } else {
                            small_heap.push(window.current, row);
                        }
                    }

                    if window_q.len() > length {
                        (window.precedent, window.precedent_idx) = window_q.pop_front().unwrap();

                        if !window.precedent.is_nan() {
                            window.observations -= 1;

                            if small_heap.remove(window.precedent_idx) {
                            } else {
                                large_heap.remove(window.precedent_idx);
                            }
                        }
                    }
                    while small_heap.heap.len() > large_heap.heap.len() + 1 {
                        if let Some((val, idx)) = small_heap.pop() {
                            large_heap.push(val, idx);
                        }
                    }

                    while large_heap.heap.len() > small_heap.heap.len() {
                        if let Some((val, idx)) = large_heap.pop() {
                            small_heap.push(val, idx);
                        }
                    }
                    if window.observations >= min_length {
                        if small_heap.heap.len() > large_heap.heap.len() {
                            if let Some((val, _)) = small_heap.peek() {
                                output_col[row] = val;
                            }
                        } else if !small_heap.heap.is_empty() {
                            let s_val: f64 = small_heap.peek().unwrap().0;
                            let l_val: f64 = large_heap.peek().unwrap().0;
                            output_col[row] = (s_val + l_val) / 2.0;
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

fn move_median_single<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        for (input_col, output_col) in input_columns.into_iter().zip(output_columns.iter_mut()) {
            let mut small_heap = Indexed::new(length, num_rows, true);
            let mut large_heap = Indexed::new(length, num_rows, false);
            let mut window_q: VecDeque<(f64, usize)> = VecDeque::with_capacity(length + 1);
            let mut window: calculators::WindowState = calculators::WindowState::new();

            for row in 0..length {
                window.current = input_col[row];

                window_q.push_back((window.current, row));

                if !window.current.is_nan() {
                    window.observations += 1;

                    if let Some((max_small, _)) = small_heap.peek() {
                        if window.current > max_small {
                            large_heap.push(window.current, row);
                        } else {
                            small_heap.push(window.current, row);
                        }
                    } else {
                        small_heap.push(window.current, row);
                    }
                }

                while small_heap.heap.len() > large_heap.heap.len() + 1 {
                    if let Some((val, idx)) = small_heap.pop() {
                        large_heap.push(val, idx);
                    }
                }

                while large_heap.heap.len() > small_heap.heap.len() {
                    if let Some((val, idx)) = large_heap.pop() {
                        small_heap.push(val, idx);
                    }
                }
                if window.observations >= min_length {
                    if small_heap.heap.len() > large_heap.heap.len() {
                        if let Some((val, _)) = small_heap.peek() {
                            output_col[row] = val;
                        }
                    } else if !small_heap.heap.is_empty() {
                        let s_val: f64 = small_heap.peek().unwrap().0;
                        let l_val: f64 = large_heap.peek().unwrap().0;
                        output_col[row] = (s_val + l_val) / 2.0;
                    }
                }
            }

            for row in length..num_rows {
                window.current = input_col[row];
                window_q.push_back((window.current, row));

                if !window.current.is_nan() {
                    window.observations += 1;

                    if let Some((max_small, _)) = small_heap.peek() {
                        if window.current > max_small {
                            large_heap.push(window.current, row);
                        } else {
                            small_heap.push(window.current, row);
                        }
                    } else {
                        small_heap.push(window.current, row);
                    }
                }

                if window_q.len() > length {
                    (window.precedent, window.precedent_idx) = window_q.pop_front().unwrap();

                    if !window.precedent.is_nan() {
                        window.observations -= 1;

                        if small_heap.remove(window.precedent_idx) {
                        } else {
                            large_heap.remove(window.precedent_idx);
                        }
                    }
                }
                while small_heap.heap.len() > large_heap.heap.len() + 1 {
                    if let Some((val, idx)) = small_heap.pop() {
                        large_heap.push(val, idx);
                    }
                }

                while large_heap.heap.len() > small_heap.heap.len() {
                    if let Some((val, idx)) = large_heap.pop() {
                        small_heap.push(val, idx);
                    }
                }
                if window.observations >= min_length {
                    if small_heap.heap.len() > large_heap.heap.len() {
                        if let Some((val, _)) = small_heap.peek() {
                            output_col[row] = val;
                        }
                    } else if !small_heap.heap.is_empty() {
                        let s_val: f64 = small_heap.peek().unwrap().0;
                        let l_val: f64 = large_heap.peek().unwrap().0;
                        output_col[row] = (s_val + l_val) / 2.0;
                    }
                }
            }
        }
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pyfunction]
pub fn agg_median<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut values: Vec<f64> = Vec::new();
                for &row in input_col.iter() {
                    if !row.is_nan() {
                        values.push(row);
                    }
                }

                if !values.is_empty() {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    let len: usize = values.len();
                    let median: f64 = if len % 2 == 0 {
                        (values[len / 2 - 1] + values[len / 2]) / 2.0
                    } else {
                        values[len / 2]
                    };
                    for val in output_col.iter_mut() {
                        *val = median;
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}
