use numpy::{ PyArray2, PyReadonlyArray2, IntoPyArray };
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use rayon::prelude::*;
use crate::calculators;

pub fn move_template<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    if parallel {
        return move_parallel::<Stat>(py, array, length, min_length);
    }

    move_single::<Stat>(py, array, length, min_length)
}

pub fn move_deque_template<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    if parallel {
        return move_deque_parallel::<Stat>(py, array, length, min_length);
    }

    move_deque_single::<Stat>(py, array, length, min_length)
}

pub fn agg_template<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    if parallel {
        return agg_parallel::<Stat>(py, array);
    }

    agg_single::<Stat>(py, array)
}

pub fn agg_deque_template<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    if parallel {
        return agg_deque_parallel::<Stat>(py, array);
    }

    agg_deque_single::<Stat>(py, array)
}

fn move_parallel<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();
    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
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
            });
    });
    Ok(output.into_pyarray(py).into())
}

fn move_single<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        for (input_col, output_col) in input_columns.into_iter().zip(output_columns.iter_mut()) {
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
    });

    Ok(output.into_pyarray(py).into())
}

fn move_deque_parallel<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut deque = Stat::new();
                let mut window = calculators::WindowState::new();

                for row in 0..length {
                    window.current = input_col[row];
                    if !window.current.is_nan() {
                        window.observations += 1;
                        Stat::add_with_index(&mut deque, window.current, row);
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
            });
    });

    Ok(output.into_pyarray(py).into())
}

fn move_deque_single<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    length: usize,
    min_length: usize
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        for (input_col, output_col) in input_columns.into_iter().zip(output_columns.iter_mut()) {
            let mut deque = Stat::new();
            let mut window = calculators::WindowState::new();

            for row in 0..length {
                window.current = input_col[row];
                if !window.current.is_nan() {
                    window.observations += 1;
                    Stat::add_with_index(&mut deque, window.current, row);
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
    });
    Ok(output.into_pyarray(py).into())
}

fn agg_parallel<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut state = Stat::new();
                let mut observation_count: usize = 0;
                for &value in input_col.iter() {
                    let current: f64 = value;
                    if !current.is_nan() {
                        observation_count += 1;
                        Stat::add_value(&mut state, current);
                    }
                }
                if observation_count > 0 {
                    let stat_result: f64 = Stat::get(&state, observation_count);
                    for val in output_col.iter_mut() {
                        *val = stat_result;
                    }
                }
            });
    });

    Ok(output.into_pyarray(py).into())
}

fn agg_single<Stat: calculators::StatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        for (input_col, output_col) in input_columns.into_iter().zip(output_columns.iter_mut()) {
            let mut state = Stat::new();
            let mut observation_count: usize = 0;
            for &value in input_col.iter() {
                let current: f64 = value;
                if !current.is_nan() {
                    observation_count += 1;
                    Stat::add_value(&mut state, current);
                }
            }
            if observation_count > 0 {
                let stat_result: f64 = Stat::get(&state, observation_count);
                for val in output_col.iter_mut() {
                    *val = stat_result;
                }
            }
        }
    });

    Ok(output.into_pyarray(py).into())
}

fn agg_deque_parallel<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        input_columns
            .into_par_iter()
            .zip(output_columns.par_iter_mut())
            .for_each(|(input_col, output_col)| {
                let mut deque = Stat::new();
                let mut observation_count: usize = 0;
                for (idx, &value) in input_col.iter().enumerate() {
                    if !value.is_nan() {
                        observation_count += 1;
                        Stat::add_with_index(&mut deque, value, idx);
                    }
                }
                if observation_count > 0 {
                    if let Some(&(val, _)) = deque.front() {
                        for output_val in output_col.iter_mut() {
                            *output_val = val;
                        }
                    }
                }
            });
    });

    Ok(output.into_pyarray(py).into())
}

fn agg_deque_single<Stat: calculators::DequeStatCalculator>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>
) -> PyResult<Py<PyArray2<f64>>> {
    let array = array.as_array();
    let (num_rows, num_cols) = array.dim();
    let mut output = Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);
    let input_columns: Vec<_> = array.columns().into_iter().collect();
    let mut output_columns: Vec<_> = output.columns_mut().into_iter().collect();

    py.allow_threads(move || {
        for (input_col, output_col) in input_columns.into_iter().zip(output_columns.iter_mut()) {
            let mut deque = Stat::new();
            let mut observation_count: usize = 0;
            for (idx, &value) in input_col.iter().enumerate() {
                if !value.is_nan() {
                    observation_count += 1;
                    Stat::add_with_index(&mut deque, value, idx);
                }
            }
            if observation_count > 0 {
                if let Some(&(val, _)) = deque.front() {
                    for output_val in output_col.iter_mut() {
                        *output_val = val;
                    }
                }
            }
        }
    });

    Ok(output.into_pyarray(py).into())
}
