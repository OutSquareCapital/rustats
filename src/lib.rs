use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use std::collections::{ VecDeque };
use rayon::prelude::*;
mod stats;
mod heaps;
mod calculators;
mod templates;

#[pyfunction]
fn move_sum<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_template::<calculators::Sum>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_mean<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_template::<calculators::Mean>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_var<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_template::<calculators::Var>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_std<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_template::<calculators::Stdev>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_skewness<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_template::<calculators::Skewness>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_kurtosis<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_template::<calculators::Kurtosis>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_min<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_deque_template::<calculators::Min>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_max<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f32>>> {
    templates::move_deque_template::<calculators::Max>(py, array, length, min_length, parallel)
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
                let mut small_heap = heaps::Indexed::new(length, num_rows, true);
                let mut large_heap = heaps::Indexed::new(length, num_rows, false);
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

#[pyfunction]
fn move_rank<'py>(
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
                for row in min_length..length {
                    let current: f32 = input_col[row];
                    if current.is_nan() {
                        continue;
                    }
                    let mut greater_count = 0;
                    let mut equal_count = 1;
                    let mut valid_count = 1;
                    for j in 0..row {
                        let other: f32 = input_col[j];
                        if !other.is_nan() {
                            valid_count += 1;
                            if current > other {
                                greater_count += 2;
                            } else if current == other {
                                equal_count += 1;
                            }
                        }
                    }

                    if valid_count >= min_length {
                        output_col[row] = stats::rank(greater_count, equal_count, valid_count);
                    }
                }
                for row in length..num_rows {
                    let current: f32 = input_col[row];
                    if current.is_nan() {
                        continue;
                    }
                    let mut greater_count = 0;
                    let mut equal_count = 1;
                    let mut valid_count = 1;
                    let start_idx: usize = row - length + 1;
                    for j in start_idx..row {
                        let other: f32 = input_col[j];
                        if !other.is_nan() {
                            valid_count += 1;
                            if current > other {
                                greater_count += 2;
                            } else if current == other {
                                equal_count += 1;
                            }
                        }
                    }

                    if valid_count >= min_length {
                        output_col[row] = stats::rank(greater_count, equal_count, valid_count);
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(move_sum, module)?)?;
    module.add_function(wrap_pyfunction!(move_std, module)?)?;
    module.add_function(wrap_pyfunction!(move_var, module)?)?;
    module.add_function(wrap_pyfunction!(move_mean, module)?)?;
    module.add_function(wrap_pyfunction!(move_max, module)?)?;
    module.add_function(wrap_pyfunction!(move_min, module)?)?;
    module.add_function(wrap_pyfunction!(move_median, module)?)?;
    module.add_function(wrap_pyfunction!(move_skewness, module)?)?;
    module.add_function(wrap_pyfunction!(move_kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(move_rank, module)?)?;
    Ok(())
}
