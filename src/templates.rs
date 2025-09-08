use crate::calculators::{DequeStatCalculator, StatCalculator, ValidCounter, WindowState};
use crate::engines::{process_1d, process_2d, Array1DOutput, Array2DOutput, WindowConfig};
use crate::heaps::IndexedProcessor;
use numpy::{ndarray as nd, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::{
    collections::VecDeque,
    ops::{Add, AddAssign, Not, Sub, SubAssign},
};

pub fn move_indexed_1d(
    py: Python<'_>,
    array: PyReadonlyArray1<'_, f64>,
    config: WindowConfig,
) -> Array1DOutput {
    process_1d(py, array, config, |input, output, config, num_rows| {
        let mut processor = IndexedProcessor::new(config.length, num_rows);
        let mut window = WindowState::new();

        (0..config.length).for_each(|row| {
            window.current = input[row];
            processor.deque.push_back((window.current, row));

            if window.current.is_nan().not() {
                window.observations.add_assign(1);
                processor.push_values(window.current, row);
            }
            finalize_indexed(&mut window, &mut processor, output, row, &config);
        });

        (config.length..num_rows).for_each(|row| {
            window.current = input[row];
            processor.deque.push_back((window.current, row));

            if window.current.is_nan().not() {
                window.observations.add_assign(1);
                processor.push_values(window.current, row);
            }

            processor.remove(&mut window, config.length);
            finalize_indexed(&mut window, &mut processor, output, row, &config);
        });
    })
}

pub fn move_indexed_2d(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
) -> Array2DOutput {
    process_2d(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut processor = IndexedProcessor::new(config.length, num_rows);
            let mut window = WindowState::new();
            (0..config.length).for_each(|row| {
                process_indexed(&mut window, input_col, &mut processor, row);
                finalize_indexed(&mut window, &mut processor, output_col, row, &config);
            });

            (config.length..num_rows).for_each(|row| {
                process_indexed(&mut window, input_col, &mut processor, row);
                processor.remove(&mut window, config.length);
                finalize_indexed(&mut window, &mut processor, output_col, row, &config);
            });
        },
    )
}

fn process_indexed(
    window: &mut WindowState,
    input_col: &nd::ArrayView1<f64>,
    processor: &mut IndexedProcessor,
    row: usize,
) {
    window.current = input_col[row];
    processor.deque.push_back((window.current, row));

    if window.current.is_nan().not() {
        window.observations.add_assign(1);
        processor.push_values(window.current, row);
    }
}

fn finalize_indexed(
    window: &mut WindowState,
    processor: &mut IndexedProcessor,
    output_col: &mut nd::ArrayViewMut1<f64>,
    row: usize,
    config: &WindowConfig,
) {
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

pub fn move_valid_count_1d(
    py: Python<'_>,
    array: PyReadonlyArray1<'_, f64>,
    config: WindowConfig,
) -> Array1DOutput {
    process_1d(py, array, config, |input, output, config, num_rows| {
        let mut rank_count = ValidCounter::new();

        (config.min_length.sub(1)..config.length).for_each(|row| {
            rank_count.reset();
            let current: f64 = input[row];
            if current.is_nan() {
                return;
            }

            (0..row).for_each(|j| {
                rank_count.add(input[j], current);
            });

            if rank_count.valid_count.ge(&config.min_length) {
                output[row] = rank_count.get();
            }
        });

        (config.length..num_rows).for_each(|row| {
            rank_count.reset();
            let current: f64 = input[row];
            if current.is_nan() {
                return;
            }
            (row.sub(config.length).add(1)..row).for_each(|j| {
                rank_count.add(input[j], current);
            });

            if rank_count.valid_count.ge(&config.min_length) {
                output[row] = rank_count.get();
            }
        });
    })
}

pub fn move_valid_count_2d(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
) -> Array2DOutput {
    process_2d(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut rank_count = ValidCounter::new();
            (config.min_length.sub(1)..config.length).for_each(|row| {
                rank_count.reset();
                let current: f64 = input_col[row];
                if current.is_nan() {
                    return;
                }

                (0..row).for_each(|j| {
                    rank_count.add(input_col[j], current);
                });

                if rank_count.valid_count.ge(&config.min_length) {
                    output_col[row] = rank_count.get();
                }
            });

            (config.length..num_rows).for_each(|row| {
                rank_count.reset();
                let current: f64 = input_col[row];
                if current.is_nan() {
                    return;
                }
                (row.sub(config.length).add(1)..row).for_each(|j| {
                    rank_count.add(input_col[j], current);
                });

                if rank_count.valid_count.ge(&config.min_length) {
                    output_col[row] = rank_count.get();
                }
            });
        },
    )
}

pub fn move_accumulator_1d<Stat: StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray1<'_, f64>,
    config: WindowConfig,
) -> Array1DOutput {
    process_1d(py, array, config, |input, output, config, num_rows| {
        let mut state = Stat::new();
        let mut window = WindowState::new();

        (0..config.length).for_each(|row| {
            window.current = input[row];
            if window.current.is_nan().not() {
                window.observations.add_assign(1);
                Stat::add_value(&mut state, window.current);
            }
            finalize_accumulator::<Stat>(&window, &state, output, row, &config);
        });

        (config.length..num_rows).for_each(|row| {
            window.current = input[row];
            window.precedent_idx = row.sub(config.length);
            window.precedent = input[window.precedent_idx];
            window.compute_row::<Stat>(&mut state);
            finalize_accumulator::<Stat>(&window, &state, output, row, &config);
        });
    })
}

pub fn move_accumulator_2d<Stat: StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
) -> Array2DOutput {
    process_2d(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut state = Stat::new();
            let mut window = WindowState::new();

            (0..config.length).for_each(|row| {
                window.current = input_col[row];
                if window.current.is_nan().not() {
                    window.observations.add_assign(1);
                    Stat::add_value(&mut state, window.current);
                }
                finalize_accumulator::<Stat>(&window, &state, output_col, row, &config);
            });

            (config.length..num_rows).for_each(|row| {
                window.refresh(&input_col, row, config.length);
                window.compute_row::<Stat>(&mut state);
                finalize_accumulator::<Stat>(&window, &state, output_col, row, &config);
            });
        },
    )
}

fn finalize_accumulator<Stat: StatCalculator>(
    window: &WindowState,
    state: &<Stat as StatCalculator>::Accumulator,
    output_col: &mut nd::ArrayViewMut1<f64>,
    row: usize,
    config: &WindowConfig,
) {
    if window.observations.ge(&config.min_length) {
        output_col[row] = Stat::get(&state, window.observations);
    }
}

pub fn move_deque_1d<Stat: DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray1<'_, f64>,
    config: WindowConfig,
) -> Array1DOutput {
    process_1d(py, array, config, |input, output, config, num_rows| {
        let mut deque = Stat::new();
        let mut window = WindowState::new();

        (0..config.length).for_each(|row| {
            window.current = input[row];
            if window.current.is_nan().not() {
                window.observations.add_assign(1);
                Stat::add_value(&mut deque, window.current, row);
            }
            finalize_deque(&window, &mut deque, output, row, &config);
        });

        (config.length..num_rows).for_each(|row| {
            window.current = input[row];
            window.precedent_idx = row.sub(config.length);
            window.precedent = input[window.precedent_idx];

            if window.precedent.is_nan().not() {
                window.observations.sub_assign(1);
                if let Some(&(_, front_idx)) = deque.front() {
                    if front_idx.eq(&window.precedent_idx) {
                        deque.pop_front();
                    }
                }
            }

            if window.current.is_nan().not() {
                window.observations.add_assign(1);
                Stat::add_value(&mut deque, window.current, row);
            }

            finalize_deque(&window, &mut deque, output, row, &config);
        });
    })
}

pub fn move_deque_2d<Stat: DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
) -> Array2DOutput {
    process_2d(
        py,
        array,
        config,
        |input_col, output_col, config, num_rows| {
            let mut deque = Stat::new();
            let mut window = WindowState::new();

            (0..config.length).for_each(|row| {
                window.current = input_col[row];
                if window.current.is_nan().not() {
                    window.observations.add_assign(1);
                    Stat::add_value(&mut deque, window.current, row);
                }
                finalize_deque(&window, &mut deque, output_col, row, &config);
            });

            (config.length..num_rows).for_each(|row| {
                window.refresh(&input_col, row, config.length);
                window.compute_deque_row::<Stat>(&mut deque, row);
                finalize_deque(&window, &mut deque, output_col, row, &config);
            });
        },
    )
}

fn finalize_deque(
    window: &WindowState,
    deque: &mut VecDeque<(f64, usize)>,
    output_col: &mut nd::ArrayViewMut1<f64>,
    row: usize,
    config: &WindowConfig,
) {
    if window.observations.ge(&config.min_length) {
        if let Some(&(val, _)) = deque.front() {
            output_col[row] = val;
        }
    }
}
