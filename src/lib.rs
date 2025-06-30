use pyo3::prelude::*;
use numpy::PyReadonlyArray2;
mod stats;
mod calculators;
mod templates;

use crate::calculators as clc;
use crate::templates as tmpl;

macro_rules! define_move_functions {
    ($(($name:ident, $processor:expr, $calculator:ty)),*) => {
        $(
            #[pyfunction]
            pub fn $name<'py>(
                py: Python<'py>,
                array: PyReadonlyArray2<'py, f64>,
                length: usize,
                min_length: usize,
                parallel: bool
            ) -> tmpl::ArrayOutput {
                $processor(py, array, length, min_length, parallel)
            }
        )*
    };
}

define_move_functions!(
    (move_sum, tmpl::move_accumulator::<clc::Sum>, clc::Sum),
    (move_mean, tmpl::move_accumulator::<clc::Mean>, clc::Mean),
    (move_var, tmpl::move_accumulator::<clc::Var>, clc::Var),
    (move_std, tmpl::move_accumulator::<clc::Stdev>, clc::Stdev),
    (move_skewness, tmpl::move_accumulator::<clc::Skewness>, clc::Skewness),
    (move_kurtosis, tmpl::move_accumulator::<clc::Kurtosis>, clc::Kurtosis),
    (move_min, tmpl::move_deque::<clc::Min>, clc::Min),
    (move_max, tmpl::move_deque::<clc::Max>, clc::Max),
    (move_median, tmpl::move_indexed, ()),
    (move_rank, tmpl::move_valid_count, ())
);

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

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(move_sum, module)?)?;
    module.add_function(wrap_pyfunction!(move_std, module)?)?;
    module.add_function(wrap_pyfunction!(move_var, module)?)?;
    module.add_function(wrap_pyfunction!(move_mean, module)?)?;
    module.add_function(wrap_pyfunction!(move_max, module)?)?;
    module.add_function(wrap_pyfunction!(move_min, module)?)?;
    module.add_function(wrap_pyfunction!(move_median, module)?)?;
    module.add_function(wrap_pyfunction!(move_skewness, module)?)?;
    module.add_function(wrap_pyfunction!(move_kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(move_rank, module)?)?;
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
