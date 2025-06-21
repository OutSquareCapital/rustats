use numpy::{ IntoPyArray, PyArray2, PyArrayLike2, PyArrayMethods };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2 };

#[pyfunction]
fn get_max<'py>(
    py: Python<'py>,
    array: PyArrayLike2<'py, f32>,
    length: usize, 
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.to_owned_array();
    let shape = array.shape();
    let num_rows: usize = shape[0];
    let num_cols: usize = shape[1];
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);

    for col in 0..num_cols {
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

        for row in length..num_rows {
            let current: f32 = array[[row, col]];
            let previous = array[[row - length, col]];
            
            if !current.is_nan() {
                observation_count += 1;
                if current > current_max {
                    current_max = current;
                }
            }
            
            if !previous.is_nan() {
                observation_count -= 1;
                if previous == current_max {
                    // Recalculate max if the previous max value is leaving the window
                    current_max = f32::NEG_INFINITY;
                    for i in (row - length + 1)..=row {
                        let val: f32 = array[[i, col]];
                        if !val.is_nan() && val > current_max {
                            current_max = val;
                        }
                    }
                }
            }
            
            if observation_count >= min_length {
                output[[row, col]] = current_max;
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