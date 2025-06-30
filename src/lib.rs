use pyo3::prelude::*;
mod stats;
mod calculators;
mod templates;
mod funcs;

#[pymodule(name = "rustats")]
fn rustats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(funcs::move_sum, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_std, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_var, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_mean, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_max, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_min, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_median, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_skewness, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::move_rank, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_sum, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_std, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_var, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_mean, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_max, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_min, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_skewness, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_kurtosis, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_median, module)?)?;
    module.add_function(wrap_pyfunction!(funcs::agg_rank, module)?)?;
    Ok(())
}
