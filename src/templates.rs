use numpy::{ PyArray2, PyReadonlyArray2, IntoPyArray };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use rayon::prelude::*;
use crate::calculators;

pub fn move_parallel<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f32>,
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
                let mut state = Stat::new();
                let mut observation_count: usize = 0;

                for row in 0..length {
                    let current: f64 = input_col[row] as f64;
                    if !current.is_nan() {
                        observation_count += 1;
                        Stat::add_value(&mut state, current);
                    }

                    if observation_count >= min_length {
                        output_col[row] = Stat::get(&state, observation_count);
                    }
                }

                for row in length..num_rows {
                    let current: f64 = input_col[row] as f64;
                    let precedent_idx: usize = row - length;
                    let precedent: f64 = input_col[precedent_idx] as f64;

                    if !current.is_nan() {
                        observation_count += 1;
                        Stat::add_value(&mut state, current);
                    }

                    if !precedent.is_nan() {
                        observation_count -= 1;
                        Stat::remove_value(&mut state, precedent);
                    }

                    if observation_count >= min_length {
                        output_col[row] = Stat::get(&state, observation_count);
                    }
                }
            });
    });

    Ok(output.into_pyarray(py).into())
}

pub fn move_single<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f32>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f32>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f32>::from_elem((num_rows, num_cols), f32::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        for (input_col, output_col) in input_columns.into_iter().zip(output_columns.iter_mut()) {
            let mut state = Stat::new();
            let mut observation_count: usize = 0;

            for row in 0..length {
                let current: f64 = input_col[row] as f64;
                if !current.is_nan() {
                    observation_count += 1;
                    Stat::add_value(&mut state, current);
                }

                if observation_count >= min_length {
                    output_col[row] = Stat::get(&state, observation_count);
                }
            }

            for row in length..num_rows {
                let current: f64 = input_col[row] as f64;
                let precedent_idx: usize = row - length;
                let precedent: f64 = input_col[precedent_idx] as f64;

                if !current.is_nan() {
                    observation_count += 1;
                    Stat::add_value(&mut state, current);
                }

                if !precedent.is_nan() {
                    observation_count -= 1;
                    Stat::remove_value(&mut state, precedent);
                }

                if observation_count >= min_length {
                    output_col[row] = Stat::get(&state, observation_count);
                }
            }
        }
    });

    Ok(output.into_pyarray(py).into())
}
