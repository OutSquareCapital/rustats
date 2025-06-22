use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use numpy::ndarray::{Array2, Axis};
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
    array: PyReadonlyArray2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let shape = array.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
        // Itérer sur les colonnes (axe 1)
        for (col_idx, col_view) in array.axis_iter(Axis(1)).enumerate() {
            // Deque stocke (valeur, index_de_ligne) pour éviter les accès mémoire redondants
            let mut max_deque: VecDeque<(f32, usize)> = VecDeque::with_capacity(length);
            let mut observation_count: usize = 0;

            for row_idx in 0..num_rows {
                // Retirer l'élément qui sort de la fenêtre
                if row_idx >= length {
                    let prev_idx = row_idx - length;
                    let prev_val = col_view[prev_idx];
                    if !prev_val.is_nan() {
                        observation_count -= 1;
                    }
                    if let Some(&(_, front_idx)) = max_deque.front() {
                        if front_idx == prev_idx {
                            max_deque.pop_front();
                        }
                    }
                }

                let current_val = col_view[row_idx];
                if !current_val.is_nan() {
                    observation_count += 1;
                    // Maintenir la propriété de la deque (valeurs décroissantes)
                    while let Some(&(back_val, _)) = max_deque.back() {
                        if back_val < current_val {
                            max_deque.pop_back();
                        } else {
                            break;
                        }
                    }
                    max_deque.push_back((current_val, row_idx));
                }

                // Assigner la sortie si la fenêtre est pleine et a assez d'observations
                if row_idx + 1 >= length && observation_count >= min_length {
                    if let Some(&(max_val, _)) = max_deque.front() {
                        output[[row_idx, col_idx]] = max_val;
                    }
                }
            }
        }
    Ok(output.into_pyarray(py).into())
}

#[pyfunction]
fn get_max_new_2<'py>(
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
                let prev: f32 = array[[prev_idx, col]];
                if !prev.is_nan() {
                    observation_count -= 1;
                }
                if let Some(&(_, front_idx)) = max_deque.front() {
                    if front_idx == prev_idx {
                        max_deque.pop_front();
                    }
                }
            }
            let current: f32 = array[[row, col]];
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
                    output[[row, col]] = max_val;
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
    module.add_function(wrap_pyfunction!(get_max_new_2, module)?)?;
    Ok(())
}
