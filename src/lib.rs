use numpy::{ IntoPyArray, PyArray2, PyArrayLike2, PyArrayMethods };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
use std::collections::VecDeque;

#[pyfunction]
fn get_max<'py>(
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
        let mut max_deque: VecDeque<usize> = VecDeque::new();
        let mut observation_count = 0;
        let mut current_max = f32::NEG_INFINITY;
        for row in 0..length.min(num_rows) {
            let current: f32 = array[[row, col]];
            if !current.is_nan() {
                observation_count += 1;
                if current > current_max {
                    current_max = current;
                }
            }
            if observation_count >= min_length {
                output[[row, col]] = current_max;
            }
        }

        for row in 0..num_rows {
            if row >= length {
                let prev_idx = row - length;
                let prev = array[[prev_idx, col]];
                if !prev.is_nan() {
                    observation_count -= 1;
                }
                if let Some(&front_idx) = max_deque.front() {
                    if front_idx == prev_idx {
                        max_deque.pop_front();
                    }
                }
            }

            let current = array[[row, col]];
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

#[pymodule(name = "rustats")]
fn rustats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_max, m)?)?;
    Ok(())
}
