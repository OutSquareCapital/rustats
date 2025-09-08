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
    ($(($name:ident, $processor_1d_f64:expr,$processor_1d_f32:expr, $processor_2d_f64:expr, $processor_2d_f32:expr, $calculator:ty)),*) => {
        $(
            #[pyfunction]
            pub fn $name<'py>(
                py: Python<'py>,
                py_array: PyObject,
                length: usize,
                min_length: usize
            ) -> PyResult<PyObject> {
                let config = WindowConfig {
                    length,
                    min_length
                };

                if let Ok(array) = py_array.extract::<PyReadonlyArray1<f64>>(py) {
                    Ok($processor_1d_f64(py, array, config)?.into())
                }
                else if let Ok(array) = py_array.extract::<PyReadonlyArray1<f32>>(py) {
                    Ok($processor_1d_f32(py, array, config)?.into())
                }
                else if let Ok(array) = py_array.extract::<PyReadonlyArray2<f64>>(py) {
                    Ok($processor_2d_f64(py, array, config)?.into())
                }
                else if let Ok(array) = py_array.extract::<PyReadonlyArray2<f32>>(py) {
                    Ok($processor_2d_f32(py, array, config)?.into())
                }
                else {
                    Err(pyo3::exceptions::PyTypeError::new_err(
                        "Input must be a 1D or 2D numpy array of float values"
                    ))
                }
            }
        )*
    };
}

define_move_functions!(
    (
        move_sum,
        tmpl::move_accumulator_1d::<clc::Sum, f64>,
        tmpl::move_accumulator_1d::<clc::Sum, f32>,
        tmpl::move_accumulator_2d::<clc::Sum, f64>,
        tmpl::move_accumulator_2d::<clc::Sum, f32>,
        clc::Sum
    ),
    (
        move_mean,
        tmpl::move_accumulator_1d::<clc::Mean, f64>,
        tmpl::move_accumulator_1d::<clc::Mean, f32>,
        tmpl::move_accumulator_2d::<clc::Mean, f64>,
        tmpl::move_accumulator_2d::<clc::Mean, f32>,
        clc::Mean
    ),
    (
        move_var,
        tmpl::move_accumulator_1d::<clc::Var, f64>,
        tmpl::move_accumulator_1d::<clc::Var, f32>,
        tmpl::move_accumulator_2d::<clc::Var, f64>,
        tmpl::move_accumulator_2d::<clc::Var, f32>,
        clc::Var
    ),
    (
        move_std,
        tmpl::move_accumulator_1d::<clc::Stdev, f64>,
        tmpl::move_accumulator_1d::<clc::Stdev, f32>,
        tmpl::move_accumulator_2d::<clc::Stdev, f64>,
        tmpl::move_accumulator_2d::<clc::Stdev, f32>,
        clc::Stdev
    ),
    (
        move_skewness,
        tmpl::move_accumulator_1d::<clc::Skewness, f64>,
        tmpl::move_accumulator_1d::<clc::Skewness, f32>,
        tmpl::move_accumulator_2d::<clc::Skewness, f64>,
        tmpl::move_accumulator_2d::<clc::Skewness, f32>,
        clc::Skewness
    ),
    (
        move_kurtosis,
        tmpl::move_accumulator_1d::<clc::Kurtosis, f64>,
        tmpl::move_accumulator_1d::<clc::Kurtosis, f32>,
        tmpl::move_accumulator_2d::<clc::Kurtosis, f64>,
        tmpl::move_accumulator_2d::<clc::Kurtosis, f32>,
        clc::Kurtosis
    ),
    (
        move_min,
        tmpl::move_deque_1d::<clc::Min, f64>,
        tmpl::move_deque_1d::<clc::Min, f32>,
        tmpl::move_deque_2d::<clc::Min, f64>,
        tmpl::move_deque_2d::<clc::Min, f32>,
        clc::Min
    ),
    (
        move_max,
        tmpl::move_deque_1d::<clc::Max, f64>,
        tmpl::move_deque_1d::<clc::Max, f32>,
        tmpl::move_deque_2d::<clc::Max, f64>,
        tmpl::move_deque_2d::<clc::Max, f32>,
        clc::Max
    ),
    (
        move_median,
        tmpl::move_indexed_1d::<f64>,
        tmpl::move_indexed_1d::<f32>,
        tmpl::move_indexed_2d::<f64>,
        tmpl::move_indexed_2d::<f32>,
        ()
    ),
    (
        move_rank,
        tmpl::move_valid_count_1d::<f64>,
        tmpl::move_valid_count_1d::<f32>,
        tmpl::move_valid_count_2d::<f64>,
        tmpl::move_valid_count_2d::<f32>,
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
