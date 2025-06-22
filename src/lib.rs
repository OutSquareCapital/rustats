use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use std::collections::{ VecDeque, BTreeSet };
use rayon::prelude::*;
use std::cmp::Ordering;

#[pyfunction]
fn move_max<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let shape: &[usize] = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
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
    let shape: &[usize] = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
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

#[derive(Debug, Copy, Clone, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[pyfunction]
fn move_median_old<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let shape: &[usize] = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                type MedianItem = (OrdF32, usize);
                let mut small_set: BTreeSet<MedianItem> = BTreeSet::new();
                let mut large_set: BTreeSet<MedianItem> = BTreeSet::new();

                let mut window_q: VecDeque<(f32, usize)> = VecDeque::with_capacity(length + 1);

                for row in 0..num_rows {
                    let current_val: f32 = input_col[row];
                    window_q.push_back((current_val, row));

                    if !current_val.is_nan() {
                        let current_item: (OrdF32, usize) = (OrdF32(current_val), row);

                        if let Some(max_small) = small_set.last() {
                            if current_item > *max_small {
                                large_set.insert(current_item);
                            } else {
                                small_set.insert(current_item);
                            }
                        } else {
                            small_set.insert(current_item);
                        }
                    }

                    if window_q.len() > length {
                        let (old_val, old_idx) = window_q.pop_front().unwrap();
                        if !old_val.is_nan() {
                            let old_item: (OrdF32, usize) = (OrdF32(old_val), old_idx);
                            if !small_set.remove(&old_item) {
                                large_set.remove(&old_item);
                            }
                        }
                    }

                    if small_set.len() > large_set.len() + 1 {
                        large_set.insert(small_set.pop_last().unwrap());
                    }
                    if large_set.len() > small_set.len() {
                        small_set.insert(large_set.pop_first().unwrap());
                    }

                    if window_q.len() >= length {
                        let observation_count: usize = small_set.len() + large_set.len();
                        if observation_count >= min_length {
                            if small_set.len() > large_set.len() {
                                output_col[row] = small_set.last().unwrap().0.0;
                            } else if !small_set.is_empty() {
                                let s_val: f32 = small_set.last().unwrap().0.0;
                                let l_val: f32 = large_set.first().unwrap().0.0;
                                output_col[row] = (s_val + l_val) / 2.0;
                            }
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pyfunction]
fn move_median<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let shape: &[usize] = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                type MedianItem = (OrdF32, usize);
                let mut small_set: BTreeSet<MedianItem> = BTreeSet::new();
                let mut large_set: BTreeSet<MedianItem> = BTreeSet::new();

                let mut window_q: VecDeque<(f32, usize)> = VecDeque::with_capacity(length + 1);

                for row in 0..num_rows {
                    let current_val: f32 = input_col[row];
                    window_q.push_back((current_val, row));

                    if !current_val.is_nan() {
                        let current_item: (OrdF32, usize) = (OrdF32(current_val), row);

                        if let Some(max_small) = small_set.last() {
                            if current_item > *max_small {
                                large_set.insert(current_item);
                            } else {
                                small_set.insert(current_item);
                            }
                        } else {
                            small_set.insert(current_item);
                        }
                    }

                    if window_q.len() > length {
                        let (old_val, old_idx) = window_q.pop_front().unwrap();
                        if !old_val.is_nan() {
                            let old_item: (OrdF32, usize) = (OrdF32(old_val), old_idx);
                            let max_small: &(OrdF32, usize) = small_set.last().unwrap();
                            if old_item > *max_small {
                                large_set.remove(&old_item);
                            } else {
                                small_set.remove(&old_item);
                            }
                        }
                    }

                    // Rebalance the sets to maintain the median property.
                    if small_set.len() > large_set.len() + 1 {
                        large_set.insert(small_set.pop_last().unwrap());
                    } else if large_set.len() > small_set.len() {
                        small_set.insert(large_set.pop_first().unwrap());
                    }

                    if window_q.len() >= length {
                        let observation_count: usize = small_set.len() + large_set.len();
                        if observation_count >= min_length {
                            if small_set.len() > large_set.len() {
                                output_col[row] = small_set.last().unwrap().0.0;
                            } else if !small_set.is_empty() {
                                let s_val: f32 = small_set.last().unwrap().0.0;
                                let l_val: f32 = large_set.first().unwrap().0.0;
                                output_col[row] = (s_val + l_val) / 2.0;
                            }
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(move_max, module)?)?;
    module.add_function(wrap_pyfunction!(move_min, module)?)?;
    module.add_function(wrap_pyfunction!(move_median_old, module)?)?;
    module.add_function(wrap_pyfunction!(move_median, module)?)?;
    Ok(())
}
