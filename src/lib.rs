use numpy::{ IntoPyArray, PyArray2, PyArrayLike2, PyArrayMethods };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
use std::collections::VecDeque;

#[pyfunction]
fn get_max_old<'py>(
    py: Python<'py>,
    array: PyArrayLike2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.to_owned_array();
    let shape: &[usize] = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);

    for col in 0..num_cols {
        let mut max_deque: VecDeque<usize> = VecDeque::with_capacity(length);
        let mut observation_count: usize = 0;
        for row in 0..num_rows {
            if row >= length {
                let prev_idx: usize = row - length;
                let prev: f32 = array[[prev_idx, col]];
                if !prev.is_nan() {
                    observation_count -= 1;
                }
                if let Some(&front_idx) = max_deque.front() {
                    if front_idx == prev_idx {
                        max_deque.pop_front();
                    }
                }
            }

            let current: f32 = array[[row, col]];
            if !current.is_nan() {
                observation_count += 1;
                while let Some(&back_idx) = max_deque.back() {
                    if array[[back_idx, col]] < current {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back(row);
            }

            if row + 1 >= length && observation_count >= min_length {
                if let Some(&max_idx) = max_deque.front() {
                    output[[row, col]] = array[[max_idx, col]];
                }
            }
        }
    }
    Ok(output.into_pyarray(py).into())
}

#[pyfunction]
fn get_max_new<'py>(
    py: Python<'py>,
    array: PyArrayLike2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.to_owned_array();
    let shape: &[usize] = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    for col in 0..num_cols {
        let mut max_deque: VecDeque<usize> = VecDeque::with_capacity(length);
        let mut observation_count: usize = 0;
        for row in 0..length {
            let current: f32 = array[[row, col]];
            if !current.is_nan() {
                observation_count += 1;
                while let Some(&back_idx) = max_deque.back() {
                    if array[[back_idx, col]] < current {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back(row);
            }

            if observation_count >= min_length {
                if let Some(&max_idx) = max_deque.front() {
                    output[[row, col]] = array[[max_idx, col]];
                }
            }
        }
        for row in length..num_rows {
            let current: f32 = array[[row, col]];
            if !current.is_nan() {
                observation_count += 1;
                while let Some(&back_idx) = max_deque.back() {
                    if array[[back_idx, col]] < current {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back(row);
            }
            let prev_idx: usize = row - length;
            let prev: f32 = array[[prev_idx, col]];
            if !prev.is_nan() {
                observation_count -= 1;
            }
            if let Some(&front_idx) = max_deque.front() {
                if front_idx == prev_idx {
                    max_deque.pop_front();
                }
            }
            if observation_count >= min_length {
                if let Some(&max_idx) = max_deque.front() {
                    output[[row, col]] = array[[max_idx, col]];
                }
            }
        }
    }
    Ok(output.into_pyarray(py).into())
}

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(get_max_old, module)?)?;
    module.add_function(wrap_pyfunction!(get_max_new, module)?)?;
    Ok(())
}
