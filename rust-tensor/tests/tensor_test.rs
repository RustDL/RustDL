use rust_tensor::tensor::{Device, Tensor};

#[test]
fn test_01() {
    let a = Tensor::new(Device::Cpu, vec![10]);
    println!("1111")
}
