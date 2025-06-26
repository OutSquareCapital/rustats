use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use rayon::prelude::*;
use crate::calculators;

#[pyfunction]
pub fn move_rank<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();
    if parallel {
        py.allow_threads(move || {
            input_columns
                .into_par_iter()
                .zip(output_columns.par_iter_mut())
                .for_each(|(input_col, output_col)| {
                    for row in min_length - 1..length {
                        let current: f64 = input_col[row];
                        if current.is_nan() {
                            continue;
                        }
                        let mut rank_count = calculators::Rank::new();
                        for j in 0..row {
                            let other: f64 = input_col[j];
                            rank_count.add(other, current);
                        }

                        if rank_count.valid_count >= min_length {
                            output_col[row] = rank_count.get();
                        }
                    }
                    for row in length..num_rows {
                        let current: f64 = input_col[row];
                        if current.is_nan() {
                            continue;
                        }
                        let mut rank_count = calculators::Rank::new();
                        let start_idx: usize = row - length + 1;
                        for j in start_idx..row {
                            let other: f64 = input_col[j];
                            rank_count.add(other, current);
                        }

                        if rank_count.valid_count >= min_length {
                            output_col[row] = rank_count.get();
                        }
                    }
                });
        });

        Ok(PyArray2::from_owned_array(py, output).into())
    } else {
        py.allow_threads(move || {
            for (input_col, output_col) in input_columns
                .into_iter()
                .zip(output_columns.iter_mut()) {
                for row in min_length - 1..length {
                    let current: f64 = input_col[row];
                    if current.is_nan() {
                        continue;
                    }
                    let mut rank_count = calculators::Rank::new();
                    for j in 0..row {
                        let other: f64 = input_col[j];
                        rank_count.add(other, current);
                    }

                    if rank_count.valid_count >= min_length {
                        output_col[row] = rank_count.get();
                    }
                }
                for row in length..num_rows {
                    let current: f64 = input_col[row];
                    if current.is_nan() {
                        continue;
                    }
                    let mut rank_count = calculators::Rank::new();
                    let start_idx: usize = row - length + 1;
                    for j in start_idx..row {
                        let other: f64 = input_col[j];
                        rank_count.add(other, current);
                    }

                    if rank_count.valid_count >= min_length {
                        output_col[row] = rank_count.get();
                    }
                }
            }
        });

        Ok(PyArray2::from_owned_array(py, output).into())
    }
}
