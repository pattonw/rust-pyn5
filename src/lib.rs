#![feature(specialization)]

extern crate n5;
#[macro_use]
extern crate pyo3;

use n5::prelude::*;
use n5::ndarray::prelude::*;
use pyo3::exceptions;
use pyo3::prelude::*;

#[pyfunction]
fn create_dataset(
    _py: Python,
    root_path: &str,
    path_name: &str,
    dimensions: Vec<i64>,
    block_size: Vec<i32>,
    dtype: &str,
) -> PyResult<()> {
    let dtype = match dtype {
        "UINT8" => DataType::UINT8,
        "UINT16" => DataType::UINT16,
        "UINT32" => DataType::UINT32,
        "UINT64" => DataType::UINT64,
        "INT8" => DataType::INT8,
        "INT16" => DataType::INT16,
        "INT32" => DataType::INT32,
        "INT64" => DataType::INT64,
        "FLOAT32" => DataType::FLOAT32,
        "FLOAT64" => DataType::FLOAT64,
        _ => return Err(exceptions::ValueError::py_err(format!(
            "Datatype {} is not supported. Please choose from {:#?}",
            dtype,
            (
                "UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16", "INT32", "INT64",
                "FLOAT32", "FLOAT64"
            )
        ))),
    };

    let n = N5Filesystem::open_or_create(root_path)?;
    if !n.exists(path_name) {
        let data_attrs = DatasetAttributes::new(
            dimensions.into(),
            block_size.into(),
            dtype,
            CompressionType::new::<compression::gzip::GzipCompression>(),
        );
        n.create_dataset(path_name, &data_attrs)?;
        Ok(())
    } else {
        Err(exceptions::ValueError::py_err(format!(
            "Dataset {} already exists!",
            path_name
        )))
    }
}

macro_rules! dataset {
    ($dataset_name:ident, $d_name:ident, $d_type:ty) => {
        #[pyclass]
        struct $dataset_name {
            n5: N5Filesystem,
            attr: DatasetAttributes,
            path: String,
        }

        #[pymethods]
        impl $dataset_name {
            #[new]
            fn __new__(
                obj: &PyRawObject,
                root_path: &str,
                path_name: &str,
                read_only: bool,
            ) -> PyResult<()> {
                Ok(obj.init({
                    if read_only {
                        let n = N5Filesystem::open(root_path).unwrap();
                        let attributes = n.get_dataset_attributes(path_name).unwrap();
                        Self {
                            n5: n,
                            attr: attributes,
                            path: path_name.to_string(),
                        }
                    } else {
                        let n = N5Filesystem::open_or_create(root_path).unwrap();
                        let attributes = n.get_dataset_attributes(path_name).unwrap();
                        Self {
                            n5: n,
                            attr: attributes,
                            path: path_name.to_string(),
                        }
                    }
                }))
            }

            #[getter]
            fn block_shape(&self) -> PyResult<(Vec<i32>)> {
                Ok(self.attr.get_block_size().iter().cloned().collect())
            }

            fn read_ndarray(
                &self,
                translation: Vec<i64>,
                dimensions: Vec<i64>,
            ) -> PyResult<Vec<$d_type>> {
                let bounding_box = BoundingBox::new(translation.into(), dimensions.into());

                let block_out = self
                    .n5
                    .read_ndarray::<$d_type>(&self.path, &self.attr, &bounding_box)
                    .unwrap();
                Ok(block_out.into_raw_vec())
            }

            fn write_block(&self, position: Vec<i64>, data: Vec<$d_type>) -> PyResult<()> {
                let block_shape = self.attr.get_block_size();
                let block_size = self.attr.get_block_num_elements();

                if block_size != data.len() {
                    Err(exceptions::ValueError::py_err(format!(
                        "Data has length {} but dataset {} has blocks with shape {:?} and size {}",
                        data.len(),
                        self.path,
                        block_shape,
                        block_size
                    )))
                } else if self.n5.exists(&self.path) {
                    let block_in = VecDataBlock::new(block_shape.into(), position.into(), data);
                    self.n5.write_block(&self.path, &self.attr, &block_in)?;
                    Ok(())
                } else {
                    Err(exceptions::ValueError::py_err(format!(
                        "Dataset {} does not exist!",
                        &self.path
                    )))
                }
            }
        }
    };
}

dataset!(DatasetUINT8, UINT8, u8);
dataset!(DatasetUINT16, UINT16, u16);
dataset!(DatasetUINT32, UINT32, u32);
dataset!(DatasetUINT64, UINT64, u64);
dataset!(DatasetINT8, INT8, i8);
dataset!(DatasetINT16, INT16, i16);
dataset!(DatasetINT32, INT32, i32);
dataset!(DatasetINT64, INT64, i64);
dataset!(DatasetFLOAT32, FLOAT32, f32);
dataset!(DatasetFLOAT64, FLOAT64, f64);

#[pymodule]
fn pyn5(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(create_dataset))?;
    m.add_class::<DatasetUINT8>()?;
    m.add_class::<DatasetUINT16>()?;
    m.add_class::<DatasetUINT32>()?;
    m.add_class::<DatasetUINT64>()?;
    m.add_class::<DatasetINT8>()?;
    m.add_class::<DatasetINT16>()?;
    m.add_class::<DatasetINT32>()?;
    m.add_class::<DatasetINT64>()?;
    m.add_class::<DatasetFLOAT32>()?;
    m.add_class::<DatasetFLOAT64>()?;

    Ok(())
}
