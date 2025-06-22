use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
use std::collections::VecDeque;
use rayon::prelude::*;

#[pyfunction]
fn get_max_old<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let shape = array.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);

    let input_columns: Vec<_> = (0..num_cols)
        .map(|i| array.column(i).to_owned())
        .collect();

    let mut output_columns: Vec<_> = output
        .columns_mut()
        .into_iter()
        .collect();

    input_columns
        .into_par_iter()
        .zip(output_columns.par_iter_mut())
        .for_each(|(input_col, output_col)| {
            let mut max_deque = VecDeque::with_capacity(length);
            let mut observation_count = 0;

            for row in 0..length {
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

                if row + 1 == length && observation_count >= min_length {
                    if let Some(&(val, _)) = max_deque.front() {
                        output_col[row] = val;
                    }
                }
            }

            for row in length..num_rows {
                let out_idx = row - length;
                let old = input_col[out_idx];
                if !old.is_nan() {
                    observation_count -= 1;
                    if let Some(&(_, pos)) = max_deque.front() {
                        if pos == out_idx {
                            max_deque.pop_front();
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

                if observation_count >= min_length {
                    if let Some(&(val, _)) = max_deque.front() {
                        output_col[row] = val;
                    }
                }
            }
        });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pyfunction]
fn get_max_new<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let shape = array.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);

    let input_columns: Vec<_> = (0..num_cols)
        .map(|i| array.column(i).to_owned())
        .collect();

    let mut output_columns: Vec<_> = output
        .columns_mut()
        .into_iter()
        .collect();

    py.allow_threads( | |{
        input_columns
        .into_par_iter()
        .zip(output_columns.par_iter_mut())
        .for_each(|(input_col, output_col)| {
            let mut max_deque = VecDeque::with_capacity(length);
            let mut observation_count = 0;

            for row in 0..length {
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

                if row + 1 == length && observation_count >= min_length {
                    if let Some(&(val, _)) = max_deque.front() {
                        output_col[row] = val;
                    }
                }
            }

            for row in length..num_rows {
                let out_idx = row - length;
                let old = input_col[out_idx];
                if !old.is_nan() {
                    observation_count -= 1;
                    if let Some(&(_, pos)) = max_deque.front() {
                        if pos == out_idx {
                            max_deque.pop_front();
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

                if observation_count >= min_length {
                    if let Some(&(val, _)) = max_deque.front() {
                        output_col[row] = val;
                    }
                }
            }
        });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(get_max_old, module)?)?;
    module.add_function(wrap_pyfunction!(get_max_new, module)?)?;
    Ok(())
}
