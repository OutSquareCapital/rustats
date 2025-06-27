use numpy::{ PyArray2, PyReadonlyArray2, IntoPyArray };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2, ArrayView1, ArrayViewMut1};
use rayon::prelude::*;
use crate::calculators;

pub fn move_template<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    if parallel {
        py.allow_threads(move || {
            input_columns
                .into_par_iter()
                .zip(output_columns.par_iter_mut())
                .for_each(|(input_col, output_col)| {
                    process_stat_column::<Stat>(
                        &input_col,
                        output_col,
                        length,
                        min_length,
                        num_rows
                    );
                });
        });
    } else {
        py.allow_threads(move || {
            for (input_col, output_col) in input_columns.iter().zip(output_columns.iter_mut()) {
                process_stat_column::<Stat>(&input_col, output_col, length, min_length, num_rows);
            }
        });
    }

    Ok(output.into_pyarray(py).into())
}

pub fn move_deque_template<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    if parallel {
        py.allow_threads(move || {
            input_columns
                .into_par_iter()
                .zip(output_columns.par_iter_mut())
                .for_each(|(input_col, output_col)| {
                    process_deque_column::<Stat>(
                        &input_col,
                        output_col,
                        length,
                        min_length,
                        num_rows
                    );
                });
        });
    } else {
        py.allow_threads(move || {
            for (input_col, output_col) in input_columns.iter().zip(output_columns.iter_mut()) {
                process_deque_column::<Stat>(&input_col, output_col, length, min_length, num_rows);
            }
        });
    }

    Ok(output.into_pyarray(py).into())
}

fn process_stat_column<Stat: calculators::StatCalculator>(
    input_col: &ArrayView1<f64>,
    output_col: &mut ArrayViewMut1<f64>,
    length: usize,
    min_length: usize,
    num_rows: usize
) {
    let mut state = Stat::new();
    let mut window = calculators::WindowState::new();

    for row in 0..length {
        window.current = input_col[row];
        if !window.current.is_nan() {
            window.observations += 1;
            Stat::add_value(&mut state, window.current);
        }

        if window.observations >= min_length {
            output_col[row] = Stat::get(&state, window.observations);
        }
    }

    for row in length..num_rows {
        window.refresh(&input_col, row, length);
        window.compute_row::<Stat>(&mut state);
        if window.observations >= min_length {
            output_col[row] = Stat::get(&state, window.observations);
        }
    }
}

fn process_deque_column<Stat: calculators::DequeStatCalculator>(
    input_col: &ArrayView1<f64>,
    output_col: &mut ArrayViewMut1<f64>,
    length: usize,
    min_length: usize,
    num_rows: usize
) {
    let mut deque = Stat::new();
    let mut window = calculators::WindowState::new();

    for row in 0..length {
        window.current = input_col[row];
        if !window.current.is_nan() {
            window.observations += 1;
            Stat::add_value(&mut deque, window.current, row);
        }
        if window.observations >= min_length {
            if let Some(&(val, _)) = deque.front() {
                output_col[row] = val;
            }
        }
    }

    for row in length..num_rows {
        window.refresh(&input_col, row, length);
        window.compute_deque_row::<Stat>(&mut deque, row);
        if window.observations >= min_length {
            if let Some(&(val, _)) = deque.front() {
                output_col[row] = val;
            }
        }
    }
}
