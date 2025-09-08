use numpy::{ndarray as nd, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

pub type Array2DOutput = PyResult<Py<PyArray2<f64>>>;
pub type Array1DOutput = PyResult<Py<PyArray1<f64>>>;

pub struct WindowConfig {
    pub length: usize,
    pub min_length: usize,
}
pub fn process_1d<F>(
    py: Python<'_>,
    array: PyReadonlyArray1<'_, f64>,
    config: WindowConfig,
    process_fn: F,
) -> Array1DOutput
where
    F: FnOnce(&nd::ArrayView1<f64>, &mut nd::ArrayViewMut1<f64>, &WindowConfig, usize)
        + Send
        + Sync,
{
    let array_view = array.as_array();
    let num_rows = array_view.len();
    let mut output = nd::Array1::<f64>::from_elem(num_rows, f64::NAN);
    py.allow_threads(|| {
        process_fn(&array_view, &mut output.view_mut(), &config, num_rows);
    });

    Ok(output.into_pyarray(py).into())
}
pub fn process_2d<F>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, f64>,
    config: WindowConfig,
    process_fn: F,
) -> Array2DOutput
where
    F: Fn(&nd::ArrayView1<f64>, &mut nd::ArrayViewMut1<f64>, &WindowConfig, usize) + Send + Sync,
{
    let array_view = array.as_array();
    let (num_rows, num_cols) = array_view.dim();
    let mut output = nd::Array2::<f64>::from_elem((num_rows, num_cols), f64::NAN);

    py.allow_threads(|| {
        array_view
            .columns()
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .zip(
                output
                    .columns_mut()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .par_iter_mut(),
            )
            .for_each(|(input_col, output_col)| {
                process_fn(&input_col, output_col, &config, num_rows);
            });
    });

    Ok(output.into_pyarray(py).into())
}
