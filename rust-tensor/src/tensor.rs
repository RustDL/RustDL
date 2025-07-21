pub enum Device {
    Cpu,
    Cuda(i32),
}

pub struct Tensor {
    pub device: Device,
    pub shape: Vec<i32>,
    data: TensorData,
}

enum TensorData {
    CpuData(Vec<u8>),
    CudaData(Vec<u8>),
}

impl Tensor {
    pub fn new(device: Device, shape: Vec<i32>) -> Tensor {
        // Tensor { device, shape }
        todo!()
    }
}
