use numpy::{ndarray as nd, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

pub type Array2DOutput<T> = PyResult<Py<PyArray2<T>>>;
pub type Array1DOutput<T> = PyResult<Py<PyArray1<T>>>;

pub struct WindowConfig {
    pub length: usize,
    pub min_length: usize,
}
pub fn process_1d<F, T: nd::NdFloat + numpy::Element>(
    py: Python<'_>,
    array: PyReadonlyArray1<'_, T>,
    config: WindowConfig,
    process_fn: F,
) -> Array1DOutput<T>
where
    F: FnOnce(&nd::ArrayView1<T>, &mut nd::ArrayViewMut1<T>, &WindowConfig, usize) + Send + Sync,
{
    let array_view = array.as_array();
    let num_rows = array_view.len();
    let mut output = nd::Array1::<T>::from_elem(num_rows, T::nan());
    py.allow_threads(|| {
        process_fn(&array_view, &mut output.view_mut(), &config, num_rows);
    });

    Ok(output.into_pyarray(py).into())
}
pub fn process_2d<F, T: nd::NdFloat + numpy::Element>(
    py: Python<'_>,
    array: PyReadonlyArray2<'_, T>,
    config: WindowConfig,
    process_fn: F,
) -> Array2DOutput<T>
where
    F: Fn(&nd::ArrayView1<T>, &mut nd::ArrayViewMut1<T>, &WindowConfig, usize) + Send + Sync,
{
    let array_view = array.as_array();
    let (num_rows, num_cols) = array_view.dim();
    let mut output = nd::Array2::<T>::from_elem((num_rows, num_cols), T::nan());

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
