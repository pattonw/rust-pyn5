#![feature(specialization)]

extern crate n5;

use n5::prelude::*;

#[macro_use]
extern crate pyo3;

use pyo3::prelude::*;

#[pyfunction]
fn read_n5(_py: Python, root_path: &str, path_name: &str, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<u8>> {
    let n = N5Filesystem::open_or_create(root_path).unwrap();

    // return an error rather than panicking to allow program calling read to handle
    // case of dataset not existing
    if !n.exists(path_name) {
        panic!(format!("Dataset {} does not exist!", path_name));
    }

    let bounding_box = BoundingBox::new(translation, dimensions);
    
    // TODO: rely on dataset attributes.json
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

#[pyfunction]
fn write_n5(_py: Python, root_path: &str, path_name: &str, translation: Vec<i64>, dimensions: Vec<i32>, data: Vec<u8>) -> PyResult<()> {
    let n = N5Filesystem::open_or_create(root_path).unwrap();
    
    // TODO: rely on dataset attributes.json if available or allow attributes to be chosen
    let data_attrs = DatasetAttributes::new(
        vec![10600, 15850, 7062],
        vec![128,128,13],
        DataType::UINT8,
        CompressionType::new::<compression::gzip::GzipCompression>(),
    );

    // NOTE: allows writing of blocks that do not allign with designated block size
    let block_in = VecDataBlock::new(
        dimensions,
        translation,
        data);

    if n.exists(path_name) {
        n.write_block(path_name, &data_attrs, &block_in)?;
    } else {
        n.create_dataset(path_name, &data_attrs)?;
        n.write_block(path_name, &data_attrs, &block_in)?;
    } 
    Ok(())
}


#[pymodinit]
fn libpyn5(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_function!(read_n5))?;
    m.add_function(wrap_function!(write_n5))?;

    Ok(())
}