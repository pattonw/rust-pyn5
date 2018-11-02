#![feature(specialization)]

extern crate n5;

use n5::prelude::*;

#[macro_use]
extern crate pyo3;

use pyo3::prelude::*;

#[pyfunction]
fn read_n5(_py: Python, root_path: &str, path_name: &str, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<u8>> {
    let n = N5Filesystem::open_or_create(root_path).unwrap();

    let bounding_box = BoundingBox::new(translation, dimensions);
    
    let data_attrs = DatasetAttributes::new(
        vec![10600, 15850, 7062],
        vec![128,128,13],
        DataType::UINT8,
        CompressionType::new::<compression::gzip::GzipCompression>(),
    );

    let block_out = n.read_ndarray::<u8>(path_name, &data_attrs, &bounding_box).unwrap();
    let raw_vec = block_out.into_raw_vec();
    Ok(raw_vec)
}


#[pymodinit]
fn pyn5(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_function!(read_n5))?;

    Ok(())
}