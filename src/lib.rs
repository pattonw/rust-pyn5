extern crate n5;

use n5::prelude::*;

#[macro_use]
extern crate cpython;

use cpython::{Python, PyResult};

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

py_module_initializer!(libpyn5, initlibpyn5, PyInit_libpyn5, |py, m | {
    try!(m.add(py, "__doc__", "This module is implemented in Rust"));
    try!(m.add(py, "read_n5", py_fn!(py, get_size(root_path: &str, path_name: &str, translation: Vec<i64>, dimensions: Vec<i64>))));
    Ok(())
});