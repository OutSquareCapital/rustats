use numpy::{ IntoPyArray, PyArray2, PyArrayLike2, PyArrayMethods };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };
use std::collections::VecDeque;

#[pyfunction]
fn get_max_deepseek<'py>(
    py: Python<'py>,
    array: PyArrayLike2<'py, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.to_owned_array();
    let shape = array.shape();
    let num_rows = shape[0];
    let num_cols = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);

    // Pré-allocation d'un ring buffer par colonne
    let mut ring_buffer = vec![(0, 0); num_cols * length];
    let mut fronts = vec![0; num_cols];
    let mut rears = vec![0; num_cols];
    let mut observation_counts = vec![0; num_cols];

    for row in 0..num_rows {
        for col in 0..num_cols {
            let col_offset = col * length;
            let front = &mut fronts[col];
            let rear = &mut rears[col];
            let obs_count = &mut observation_counts[col];
            let current_val = array[[row, col]];
            
            // Gestion de la fenêtre glissante
            if row >= length {
                let prev_val = array[[row - length, col]];
                if !prev_val.is_nan() {
                    *obs_count -= 1;
                }
                
                // Vérifie si le max sortant est en tête
                if *obs_count > 0 && ring_buffer[col_offset + *front].1 == row - length {
                    *front = (*front + 1) % length;
                }
            }

            // Ajout de la nouvelle valeur
            if !current_val.is_nan() {
                *obs_count += 1;
                
                // Suppression des éléments plus petits
                while *obs_count > 1 && *rear != *front {
                    let last_index = (length + *rear - 1) % length;
                    let last_val = array[[ring_buffer[col_offset + last_index].0, col]];
                    
                    if last_val < current_val {
                        *rear = last_index;
                    } else {
                        break;
                    }
                }
                
                // Ajout au buffer
                ring_buffer[col_offset + *rear] = (row, row);
                *rear = (*rear + 1) % length;
            }

            // Écriture du résultat
            if row >= length - 1 && *obs_count >= min_length && *front != *rear {
                let max_index = ring_buffer[col_offset + *front].0;
                output[[row, col]] = array[[max_index, col]];
            }
        }
    }

    Ok(output.into_pyarray(py).into())
}

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

#[pymodule(name = "rustats")]
fn rustats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_max, m)?)?;
    m.add_function(wrap_pyfunction!(get_max_deepseek, m)?)?;
    Ok(())
}
