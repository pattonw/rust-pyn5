#![feature(specialization)]

extern crate n5;
#[macro_use]
extern crate pyo3;

use n5::prelude::*;
use pyo3::prelude::*;

#[pyclass]
struct Dataset {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl Dataset {
    #[new]
    fn __new__(obj: &PyRawObject, root_path: &str, path_name: &str) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
        obj.init(|_| {
            let n = N5Filesystem::open_or_create(root_path).unwrap();
            let attributes = n.get_dataset_attributes(path_name).unwrap();
            Dataset {
                n5: n,
                attr: attributes,
                path: path_name.to_string(),
            }
        })
    }

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<u8>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<u8>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<u8>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exc::ValueError::new(format!(
                "Data has length {} but dataset {} has blocks with shape {:?} and size {}",
                data.len(),
                self.path,
                block_shape,
                block_size
            )))
        } else if self.n5.exists(&self.path) {
            let block_in = VecDataBlock::new(block_shape, position, data);
            self.n5.write_block(&self.path, &self.attr, &block_in)?;
            Ok(())
        } else {
            Err(exc::ValueError::new(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyfunction]
fn create_dataset(
    _py: Python,
    root_path: &str,
    path_name: &str,
    dimensions: Vec<i64>,
    block_size: Vec<i32>,
) -> PyResult<()> {
    let n = N5Filesystem::open_or_create(root_path).unwrap();
    if !n.exists(path_name) {
        let data_attrs = DatasetAttributes::new(
            dimensions,
            block_size,
            DataType::UINT8,
            CompressionType::new::<compression::gzip::GzipCompression>(),
        );
        n.create_dataset(path_name, &data_attrs)?;
        Ok(())
    } else {
        Err(exc::ValueError::new(format!(
            "Dataset {} already exists!",
            path_name
        )))
    }
}

#[pymodinit]
fn libpyn5(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_function!(create_dataset))?;
    m.add_class::<Dataset>()?;

    Ok(())
}
