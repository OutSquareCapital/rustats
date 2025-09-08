use crate::calculators as clc;
use numpy::{ndarray as nd, IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Not, Sub};

pub type ArrayOutput = PyResult<Py<PyArray2<f64>>>;

pub struct WindowConfig {
    pub length: usize,
    pub min_length: usize,
    pub parallel: bool,
}

fn process_with_strategy<F>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
    process_fn: F,
) -> ArrayOutput
where
    F: Fn(&nd::ArrayView1<f64>, &mut nd::ArrayViewMut1<f64>, &WindowConfig, usize) + Send + Sync,
{
    let array_view = array.as_array();
    let (num_rows, num_cols) = array_view.dim();
    let mut output = nd::Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array_view.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        if config.parallel {
            input_columns
                .into_par_iter()
                .zip(output_columns.par_iter_mut())
                .for_each(|(input_col, output_col)| {
                    process_fn(&input_col, output_col, &config, num_rows);
                });
        } else {
            input_columns
                .iter()
                .zip(output_columns.iter_mut())
                .for_each(|(input_col, output_col)| {
                    process_fn(&input_col, output_col, &config, num_rows);
                });
        }
    });

    Ok(output.into_pyarray(py).into())
}

pub fn move_indexed<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    config: WindowConfig,
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut processor = clc::IndexedProcessor::new(config.length, num_rows);
            let mut window = clc::WindowState::new();

            for row in 0..config.length {
                window.current = input_col[row];

                processor.deque.push_back((window.current, row));

                if window.current.is_nan().not() {
                    window.observations.add_assign(1);
                    processor.push_values(window.current, row);
                }

                processor.equilibrate();

                if window.observations.ge(&config.min_length) {
                    if processor.check() {
                        if let Some((val, _)) = processor.small_heap.peek() {
                            output_col[row] = val;
                        }
                    } else if processor.small_heap.heap.is_empty().not() {
                        output_col[row] = processor.get();
                    }
                }
            }
            for row in config.length..num_rows {
                window.current = input_col[row];
                processor.deque.push_back((window.current, row));

                if window.current.is_nan().not() {
                    window.observations.add_assign(1);
                    processor.push_values(window.current, row);
                }
                processor.remove(&mut window, config.length);
                processor.equilibrate();

                if window.observations.ge(&config.min_length) {
                    if processor.check() {
                        if let Some((val, _)) = processor.small_heap.peek() {
                            output_col[row] = val;
                        }
                    } else if processor.small_heap.heap.is_empty().not() {
                        output_col[row] = processor.get();
                    }
                }
            }
        },
    )
}

pub fn move_valid_count<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    config: WindowConfig,
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut rank_count = clc::ValidCounter::new();
            for row in config.min_length.sub(1)..config.length {
                rank_count.reset();
                let current: f64 = input_col[row];
                if current.is_nan() {
                    continue;
                }

                for j in 0..row {
                    rank_count.add(input_col[j], current);
                }

                if rank_count.valid_count.ge(&config.min_length) {
                    output_col[row] = rank_count.get();
                }
            }

            for row in config.length..num_rows {
                rank_count.reset();
                let current: f64 = input_col[row];
                if current.is_nan() {
                    continue;
                }
                for j in row.sub(config.length).add(1)..row {
                    rank_count.add(input_col[j], current);
                }

                if rank_count.valid_count.ge(&config.min_length) {
                    output_col[row] = rank_count.get();
                }
            }
        },
    )
}

pub fn move_accumulator<Stat: clc::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut state = Stat::new();
            let mut window = clc::WindowState::new();

            for row in 0..config.length {
                window.current = input_col[row];
                if window.current.is_nan().not() {
                    window.observations.add_assign(1);
                    Stat::add_value(&mut state, window.current);
                }

                if window.observations.ge(&config.min_length) {
                    output_col[row] = Stat::get(&state, window.observations);
                }
            }

            for row in config.length..num_rows {
                window.refresh(&input_col, row, config.length);
                window.compute_row::<Stat>(&mut state);
                if window.observations.ge(&config.min_length) {
                    output_col[row] = Stat::get(&state, window.observations);
                }
            }
        },
    )
}

pub fn move_deque<Stat: clc::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
) -> ArrayOutput {
    process_with_strategy(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut deque = Stat::new();
            let mut window = clc::WindowState::new();

            for row in 0..config.length {
                window.current = input_col[row];
                if !window.current.is_nan() {
                    window.observations.add_assign(1);
                    Stat::add_value(&mut deque, window.current, row);
                }
                if window.observations.ge(&config.min_length) {
                    if let Some(&(val, _)) = deque.front() {
                        output_col[row] = val;
                    }
                }
            }

            for row in config.length..num_rows {
                window.refresh(&input_col, row, config.length);
                window.compute_deque_row::<Stat>(&mut deque, row);
                if window.observations.ge(&config.min_length) {
                    if let Some(&(val, _)) = deque.front() {
                        output_col[row] = val;
                    }
                }
            }
        },
    )
}
