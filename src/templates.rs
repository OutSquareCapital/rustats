use numpy::{ PyArray2, PyReadonlyArray2, IntoPyArray, ndarray as nd };
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::calculators as clc;

pub type ArrayOutput = PyResult<Py<PyArray2<f64>>>;

fn process_with_strategy<F>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool,
    process_fn: F
) -> ArrayOutput
    where
        F: Fn(&nd::ArrayView1<f64>, &mut nd::ArrayViewMut1<f64>, usize, usize, usize) + Send + Sync
{
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = nd::Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        if parallel {
            input_columns
                .into_par_iter()
                .zip(output_columns.par_iter_mut())
                .for_each(|(input_col, output_col)| {
                    process_fn(&input_col, output_col, length, min_length, num_rows);
                });
        } else {
            for (input_col, output_col) in input_columns.iter().zip(output_columns.iter_mut()) {
                process_fn(&input_col, output_col, length, min_length, num_rows);
            }
        }
    });

    Ok(output.into_pyarray(py).into())
}

pub fn move_indexed<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        length,
        min_length,
        parallel,
        |input_col, output_col, length, min_length, num_rows| {
            let mut dl = clc::DancingLinks::with_capacity(length + 1);
            let window_data = input_col.slice(nd::s![0..length]);
            dl.fill(window_data.as_slice().unwrap());

            for row in 0..length {
                if dl.len() >= min_length {
                    output_col[row] = dl.median();
                }
            }

            for row in length..num_rows {
                let oldest_idx = row - length;
                let oldest_val = input_col[oldest_idx];
                let newest_val = input_col[row];
                if !oldest_val.is_nan() {
                    dl.remove_by_original_index(oldest_idx);
                }

                if !newest_val.is_nan() {
                    dl.add_with_index(newest_val, row);
                }

                if dl.len() >= min_length {
                    output_col[row] = dl.median();
                }
            }
        }
    )
}

pub fn move_valid_count<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        length,
        min_length,
        parallel,
        |input_col, output_col, length, min_length, num_rows| {
            for row in min_length - 1..length {
                let current: f64 = input_col[row];
                if current.is_nan() {
                    continue;
                }

                let mut rank_count = clc::ValidCounter::new();
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

                let mut rank_count = clc::ValidCounter::new();
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
    )
}

pub fn move_accumulator<Stat: clc::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        length,
        min_length,
        parallel,
        |input_col, output_col, length, min_length, num_rows| {
            {
                let mut state = Stat::new();
                let mut window = clc::WindowState::new();

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
        }
    )
}

pub fn move_deque<Stat: clc::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        length,
        min_length,
        parallel,
        |input_col, output_col, length, min_length, num_rows| {
            {
                let mut deque = Stat::new();
                let mut window = clc::WindowState::new();

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
        }
    )
}
