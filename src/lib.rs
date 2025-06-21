use numpy::{ IntoPyArray, PyArray2, PyArrayLike2, PyArrayMethods };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
use std::collections::VecDeque;
use rayon::prelude::*;

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
    let shape = array.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];

    // Calculer chaque colonne en parallèle et retourner un Vec contenant le résultat de chaque colonne.
    let columns: Vec<Vec<f32>> = (0..num_cols)
        .into_par_iter()
        .map(|col| {
            let mut col_result = vec![f32::NAN; num_rows];
            let mut max_deque: std::collections::VecDeque<usize> = std::collections::VecDeque::with_capacity(length);
            let mut observation_count = 0;
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
                        col_result[row] = array[[max_idx, col]];
                    }
                }
            }
            col_result
        })
        .collect();

    // Recoller les colonnes dans une Array2.
    let mut output = ndarray::Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    for col in 0..num_cols {
        for row in 0..num_rows {
            output[[row, col]] = columns[col][row];
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
