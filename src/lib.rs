use numpy::{ PyArray2, PyReadonlyArray2 };
use pyo3::prelude::*;
mod stats;
mod calculators;
mod templates;

#[pyfunction]
fn move_sum<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_template::<calculators::Sum>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_mean<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_template::<calculators::Mean>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_var<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_template::<calculators::Var>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_std<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_template::<calculators::Stdev>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_skewness<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_template::<calculators::Skewness>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_kurtosis<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_template::<calculators::Kurtosis>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_min<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_deque_template::<calculators::Min>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn move_max<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> PyResult<Py<PyArray2<f64>>> {
    templates::move_deque_template::<calculators::Max>(py, array, length, min_length, parallel)
}

#[pyfunction]
fn agg_sum<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_mean<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_var<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_std<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_skewness<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_kurtosis<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_min<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_max<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_median<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pyfunction]
fn agg_rank<'py>() -> PyResult<Py<PyArray2<f64>>> {
    todo!()
}

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(move_sum, module)?)?;
    module.add_function(wrap_pyfunction!(move_std, module)?)?;
    module.add_function(wrap_pyfunction!(move_var, module)?)?;
    module.add_function(wrap_pyfunction!(move_mean, module)?)?;
    module.add_function(wrap_pyfunction!(move_max, module)?)?;
    module.add_function(wrap_pyfunction!(move_min, module)?)?;
    module.add_function(wrap_pyfunction!(templates::move_median, module)?)?;
    module.add_function(wrap_pyfunction!(move_skewness, module)?)?;
    module.add_function(wrap_pyfunction!(move_kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(templates::move_rank, module)?)?;
    module.add_function(wrap_pyfunction!(agg_sum, module)?)?;
    module.add_function(wrap_pyfunction!(agg_std, module)?)?;
    module.add_function(wrap_pyfunction!(agg_var, module)?)?;
    module.add_function(wrap_pyfunction!(agg_mean, module)?)?;
    module.add_function(wrap_pyfunction!(agg_max, module)?)?;
    module.add_function(wrap_pyfunction!(agg_min, module)?)?;
    module.add_function(wrap_pyfunction!(agg_skewness, module)?)?;
    module.add_function(wrap_pyfunction!(agg_kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(agg_median, module)?)?;
    module.add_function(wrap_pyfunction!(agg_rank, module)?)?;
    Ok(())
}
