use numpy::{ PyArray2, PyReadonlyArray2, IntoPyArray };
use pyo3::prelude::*;
use numpy::ndarray::{ Array2, ArrayView1, ArrayViewMut1 };
use rayon::prelude::*;
use crate::calculators;
use std::collections::VecDeque;

#[pyfunction]
pub fn move_median<'py>(
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
                    process_median_column(&input_col, output_col, length, min_length, num_rows);
                });
        });
    } else {
        py.allow_threads(move || {
            for (input_col, output_col) in input_columns.iter().zip(output_columns.iter_mut()) {
                process_median_column(&input_col, output_col, length, min_length, num_rows);
            }
        });
    }

    Ok(PyArray2::from_owned_array(py, output).into())
}

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
                    process_rank_column(&input_col, output_col, length, min_length, num_rows);
                });
        });
    } else {
        py.allow_threads(move || {
            for (input_col, output_col) in input_columns.iter().zip(output_columns.iter_mut()) {
                process_rank_column(&input_col, output_col, length, min_length, num_rows);
            }
        });
    }

    Ok(PyArray2::from_owned_array(py, output).into())
}

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

pub fn process_rank_column(
    input_col: &ArrayView1<f64>,
    output_col: &mut ArrayViewMut1<f64>,
    length: usize,
    min_length: usize,
    num_rows: usize
) {
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

fn process_median_column(
    input_col: &ArrayView1<f64>,
    output_col: &mut ArrayViewMut1<f64>,
    length: usize,
    min_length: usize,
    num_rows: usize
) {
    let mut small_heap = calculators::Indexed::new(length, num_rows, true);
    let mut large_heap = calculators::Indexed::new(length, num_rows, false);
    let mut window_q = VecDeque::with_capacity(length + 1);
    let mut window = calculators::WindowState::new();

    for row in 0..length {
        window.current = input_col[row];

        window_q.push_back((window.current, row));

        if !window.current.is_nan() {
            window.observations += 1;

            if let Some((max_small, _)) = small_heap.peek() {
                if window.current > max_small {
                    large_heap.push(window.current, row);
                } else {
                    small_heap.push(window.current, row);
                }
            } else {
                small_heap.push(window.current, row);
            }
        }

        while small_heap.heap.len() > large_heap.heap.len() + 1 {
            if let Some((val, idx)) = small_heap.pop() {
                large_heap.push(val, idx);
            }
        }

        while large_heap.heap.len() > small_heap.heap.len() {
            if let Some((val, idx)) = large_heap.pop() {
                small_heap.push(val, idx);
            }
        }

        if window.observations >= min_length {
            if small_heap.heap.len() > large_heap.heap.len() {
                if let Some((val, _)) = small_heap.peek() {
                    output_col[row] = val;
                }
            } else if !small_heap.heap.is_empty() {
                let s_val: f64 = small_heap.peek().unwrap().0;
                let l_val: f64 = large_heap.peek().unwrap().0;
                output_col[row] = (s_val + l_val) / 2.0;
            }
        }
    }

    for row in length..num_rows {
        window.current = input_col[row];
        window_q.push_back((window.current, row));

        if !window.current.is_nan() {
            window.observations += 1;

            if let Some((max_small, _)) = small_heap.peek() {
                if window.current > max_small {
                    large_heap.push(window.current, row);
                } else {
                    small_heap.push(window.current, row);
                }
            } else {
                small_heap.push(window.current, row);
            }
        }

        if window_q.len() > length {
            (window.precedent, window.precedent_idx) = window_q.pop_front().unwrap();

            if !window.precedent.is_nan() {
                window.observations -= 1;

                if small_heap.remove(window.precedent_idx) {
                } else {
                    large_heap.remove(window.precedent_idx);
                }
            }
        }

        while small_heap.heap.len() > large_heap.heap.len() + 1 {
            if let Some((val, idx)) = small_heap.pop() {
                large_heap.push(val, idx);
            }
        }

        while large_heap.heap.len() > small_heap.heap.len() {
            if let Some((val, idx)) = large_heap.pop() {
                small_heap.push(val, idx);
            }
        }

        if window.observations >= min_length {
            if small_heap.heap.len() > large_heap.heap.len() {
                if let Some((val, _)) = small_heap.peek() {
                    output_col[row] = val;
                }
            } else if !small_heap.heap.is_empty() {
                let s_val: f64 = small_heap.peek().unwrap().0;
                let l_val: f64 = large_heap.peek().unwrap().0;
                output_col[row] = (s_val + l_val) / 2.0;
            }
        }
    }
}
