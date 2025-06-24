use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use rayon::prelude::*;
use std::cmp::Ordering;
use crate::calculators;
use crate::stats;

#[pyfunction]
pub fn move_rank<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
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
                for row in min_length..length {
                    let current: f32 = input_col[row];
                    if current.is_nan() {
                        continue;
                    }
                    let mut rank_count = calculators::Rank::new();
                    for j in 0..row {
                        let other: f32 = input_col[j];
                        rank_count.add(other, current);
                    }

                    if rank_count.valid_count >= min_length {
                        output_col[row] = rank_count.get();
                    }
                }
                for row in length..num_rows {
                    let current: f32 = input_col[row];
                    if current.is_nan() {
                        continue;
                    }
                    let mut rank_count = calculators::Rank::new();
                    let start_idx: usize = row - length + 1;
                    for j in start_idx..row {
                        let other: f32 = input_col[j];
                        rank_count.add(other, current);
                    }

                    if rank_count.valid_count >= min_length {
                        output_col[row] = rank_count.get();
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}

#[pyfunction]
pub fn agg_rank<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>
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
                let mut value_indices: Vec<(f32, usize)> = Vec::new();
                for (idx, &val) in input_col.iter().enumerate() {
                    if !val.is_nan() {
                        value_indices.push((val, idx));
                    }
                }

                if !value_indices.is_empty() {
                    value_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

                    let valid_count: usize = value_indices.len();
                    let mut ranks = vec![0.0f32; num_rows];
                    let mut greater_count: usize = 0;
                    while greater_count < valid_count {
                        let value: f32 = value_indices[greater_count].0;
                        let mut j: usize = greater_count + 1;
                        while j < valid_count && value_indices[j].0 == value {
                            j += 1;
                        }
                        let rank_value: f32 = stats::rank(
                            greater_count,
                            j - greater_count,
                            valid_count
                        );
                        for k in greater_count..j {
                            let orig_idx: usize = value_indices[k].1;
                            ranks[orig_idx] = rank_value;
                        }

                        greater_count = j;
                    }
                    for (idx, val) in ranks.iter().enumerate() {
                        if !input_col[idx].is_nan() {
                            output_col[idx] = *val;
                        }
                    }
                }
            });
    });

    Ok(PyArray2::from_owned_array(py, output).into())
}
