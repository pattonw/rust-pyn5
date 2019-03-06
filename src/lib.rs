#![feature(specialization)]

extern crate n5;
#[macro_use]
extern crate pyo3;

use n5::prelude::*;
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
    let n = N5Filesystem::open_or_create(root_path).unwrap();
    if !n.exists(path_name) {
        match dtype {
            "u8" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::UINT8,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "u16" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::UINT16,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "u32" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::UINT32,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "u64" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::UINT64,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "i8" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::INT8,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "i16" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::INT16,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "i32" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::INT32,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "i64" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::INT64,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "f32" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::FLOAT32,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            "f64" => {
                let data_attrs = DatasetAttributes::new(
                    dimensions,
                    block_size,
                    DataType::FLOAT64,
                    CompressionType::new::<compression::gzip::GzipCompression>(),
                );
                n.create_dataset(path_name, &data_attrs)?;
                Ok(())
            }
            _ => Err(exceptions::ValueError::py_err(format!(
                "Datatype {} is not supported. Please choose from {:#?}",
                dtype,
                ("u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64")
            ))),
        }
    } else {
        Err(exceptions::ValueError::py_err(format!(
            "Dataset {} already exists!",
            path_name
        )))
    }
}

#[pyclass]
struct DatasetU8 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetU8 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetU16 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetU16 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<u16>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<u16>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<u16>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetU32 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetU32 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<u32>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<u32>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<u32>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetU64 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetU64 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<u64>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<u64>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<u64>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetI8 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetI8 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<i8>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<i8>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<i8>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetI16 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetI16 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<i16>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<i16>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<i16>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetI32 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetI32 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<i32>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<i32>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<i32>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetI64 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetI64 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<i64>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<i64>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<i64>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetF32 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetF32 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<f32>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<f32>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<f32>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pyclass]
struct DatasetF64 {
    n5: N5Filesystem,
    attr: DatasetAttributes,
    path: String,
}

#[pymethods]
impl DatasetF64 {
    #[new]
    fn __new__(
        obj: &PyRawObject,
        root_path: &str,
        path_name: &str,
        read_only: bool,
    ) -> PyResult<()> {
        // TODO: pass in optional attributes which can be used to create datasets rather
        // than panicing when dataset does not exist
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

    fn read_ndarray(&self, translation: Vec<i64>, dimensions: Vec<i64>) -> PyResult<Vec<f64>> {
        let bounding_box = BoundingBox::new(translation, dimensions);

        let block_out = self
            .n5
            .read_ndarray::<f64>(&self.path, &self.attr, &bounding_box)
            .unwrap();
        Ok(block_out.into_raw_vec())
    }

    fn write_block(&self, position: Vec<i64>, data: Vec<f64>) -> PyResult<()> {
        let block_shape = self.attr.get_block_size().to_vec();
        let block_size = block_shape.iter().fold(1, |a, &b| a * b) as usize;

        if block_size != data.len() {
            Err(exceptions::ValueError::py_err(format!(
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
            Err(exceptions::ValueError::py_err(format!(
                "Dataset {} does not exist!",
                &self.path
            )))
        }
    }
}

#[pymodule]
fn libpyn5(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(create_dataset))?;
    m.add_class::<DatasetU8>()?;
    m.add_class::<DatasetU16>()?;
    m.add_class::<DatasetU32>()?;
    m.add_class::<DatasetU64>()?;
    m.add_class::<DatasetI8>()?;
    m.add_class::<DatasetI16>()?;
    m.add_class::<DatasetI32>()?;
    m.add_class::<DatasetI64>()?;
    m.add_class::<DatasetF32>()?;
    m.add_class::<DatasetF64>()?;

    Ok(())
}
