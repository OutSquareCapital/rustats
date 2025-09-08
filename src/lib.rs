use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
mod accumulators;
mod calculators;
mod engines;
mod heaps;
mod stats;
mod templates;
use crate::calculators as clc;
use crate::engines::WindowConfig;
use crate::templates as tmpl;

macro_rules! define_move_functions {
    ($(($name:ident, $processor_1d:expr, $processor_2d:expr, $calculator:ty)),*) => {
        $(
            #[pyfunction]
            pub fn $name<'py>(
                py: Python<'py>,
                array: PyObject,
                length: usize,
                min_length: usize
            ) -> PyResult<PyObject> {
                let config = WindowConfig {
                    length,
                    min_length
                };

                if let Ok(array_1d) = array.extract::<PyReadonlyArray1<f64>>(py) {
                    Ok($processor_1d(py, array_1d, config)?.into())
                }
                else if let Ok(array_2d) = array.extract::<PyReadonlyArray2<f64>>(py) {
                    Ok($processor_2d(py, array_2d, config)?.into())
                }
                else {
                    Err(pyo3::exceptions::PyTypeError::new_err(
                        "Input must be a 1D or 2D numpy array of float64 values"
                    ))
                }
            }
        )*
    };
}

define_move_functions!(
    (
        move_sum,
        tmpl::move_accumulator_1d::<clc::Sum>,
        tmpl::move_accumulator_2d::<clc::Sum>,
        clc::Sum
    ),
    (
        move_mean,
        tmpl::move_accumulator_1d::<clc::Mean>,
        tmpl::move_accumulator_2d::<clc::Mean>,
        clc::Mean
    ),
    (
        move_var,
        tmpl::move_accumulator_1d::<clc::Var>,
        tmpl::move_accumulator_2d::<clc::Var>,
        clc::Var
    ),
    (
        move_std,
        tmpl::move_accumulator_1d::<clc::Stdev>,
        tmpl::move_accumulator_2d::<clc::Stdev>,
        clc::Stdev
    ),
    (
        move_skewness,
        tmpl::move_accumulator_1d::<clc::Skewness>,
        tmpl::move_accumulator_2d::<clc::Skewness>,
        clc::Skewness
    ),
    (
        move_kurtosis,
        tmpl::move_accumulator_1d::<clc::Kurtosis>,
        tmpl::move_accumulator_2d::<clc::Kurtosis>,
        clc::Kurtosis
    ),
    (
        move_min,
        tmpl::move_deque_1d::<clc::Min>,
        tmpl::move_deque_2d::<clc::Min>,
        clc::Min
    ),
    (
        move_max,
        tmpl::move_deque_1d::<clc::Max>,
        tmpl::move_deque_2d::<clc::Max>,
        clc::Max
    ),
    (
        move_median,
        tmpl::move_indexed_1d,
        tmpl::move_indexed_2d,
        ()
    ),
    (
        move_rank,
        tmpl::move_valid_count_1d,
        tmpl::move_valid_count_2d,
        ()
    )
);

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
    Ok(())
}
