use numpy::{ IntoPyArray, PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
use std::collections::VecDeque;

#[pyfunction]
fn get_max_old<'py>(
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

    for col in 0..num_cols {
        let mut max_deque: VecDeque<(f32, usize)> = VecDeque::with_capacity(length);
        let mut observation_count: usize = 0;
        for row in 0..num_rows {
            if row >= length {
                let prev_idx: usize = row - length;
                // SAFETY: The loop logic ensures prev_idx is always in bounds.
                let prev: f32 = unsafe { *array.uget((prev_idx, col)) };
                if !prev.is_nan() {
                    observation_count -= 1;
                }
                if let Some(&(_, front_idx)) = max_deque.front() {
                    if front_idx == prev_idx {
                        max_deque.pop_front();
                    }
                }
            }
            // SAFETY: The loop logic ensures row and col are always in bounds.
            let current: f32 = unsafe { *array.uget((row, col)) };
            if !current.is_nan() {
                observation_count += 1;
                while let Some(&(back_val, _)) = max_deque.back() {
                    if back_val < current {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((current, row));
            }

            if row + 1 >= length && observation_count >= min_length {
                if let Some(&(max_val, _)) = max_deque.front() {
                    // SAFETY: The loop logic ensures row and col are always in bounds.
                    unsafe {
                        *output.uget_mut((row, col)) = max_val;
                    }
                }
            }
        }
    }
    Ok(output.into_pyarray(py).into())
}

#[pyfunction]
fn get_max_new<'py>(
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

    for col in 0..num_cols {
        let mut max_deque: VecDeque<(f32, usize)> = VecDeque::with_capacity(length);
        let mut observation_count: usize = 0;

        for row in 0..length {
            // SAFETY: La logique de la boucle garantit que row et col sont valides.
            let current: f32 = unsafe { *array.uget((row, col)) };
            if !current.is_nan() {
                observation_count += 1;
                while let Some(&(back_val, _)) = max_deque.back() {
                    if back_val < current {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((current, row));
            }

            if row + 1 == length && observation_count >= min_length {
                if let Some(&(max_val, _)) = max_deque.front() {
                    // SAFETY: La logique de la boucle garantit que row et col sont valides.
                    unsafe {
                        *output.uget_mut((row, col)) = max_val;
                    }
                }
            }
        }
        for row in length..num_rows {
            let prev_idx: usize = row - length;
            // SAFETY: La logique de la boucle garantit que prev_idx est valide.
            let prev: f32 = unsafe { *array.uget((prev_idx, col)) };
            if !prev.is_nan() {
                observation_count -= 1;
            }
            if let Some(&(_, front_idx)) = max_deque.front() {
                if front_idx == prev_idx {
                    max_deque.pop_front();
                }
            }
            // SAFETY: La logique de la boucle garantit que row et col sont valides.
            let current: f32 = unsafe { *array.uget((row, col)) };
            if !current.is_nan() {
                observation_count += 1;
                while let Some(&(back_val, _)) = max_deque.back() {
                    if back_val < current {
                        max_deque.pop_back();
                    } else {
                        break;
                    }
                }
                max_deque.push_back((current, row));
            }
            if observation_count >= min_length {
                if let Some(&(max_val, _)) = max_deque.front() {
                    // SAFETY: La logique de la boucle garantit que row et col sont valides.
                    unsafe {
                        *output.uget_mut((row, col)) = max_val;
                    }
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
