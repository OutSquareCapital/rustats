use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use crate::calculators as clc;
use crate::templates as tmpl;

#[pyfunction]
pub fn move_sum<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_accumulator::<clc::Sum>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_mean<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_accumulator::<clc::Mean>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_var<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_accumulator::<clc::Var>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_std<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_accumulator::<clc::Stdev>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_skewness<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_accumulator::<clc::Skewness>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_kurtosis<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_accumulator::<clc::Kurtosis>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_min<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_deque::<clc::Min>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_max<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_deque::<clc::Max>(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_median<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_indexed(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn move_rank<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f64>,
    length: usize,
    min_length: usize,
    parallel: bool
) -> tmpl::ArrayOutput {
    tmpl::move_valid_count(py, array, length, min_length, parallel)
}

#[pyfunction]
pub fn agg_sum<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_mean<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_var<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_std<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_skewness<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_kurtosis<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_min<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_max<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_median<'py>() -> tmpl::ArrayOutput {
    todo!()
}

#[pyfunction]
pub fn agg_rank<'py>() -> tmpl::ArrayOutput {
    todo!()
}
