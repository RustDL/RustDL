#[cfg(test)]
mod test_cuda {
    use crate::{CudaErrorWrapper, cuda, nvvm};
    use std::os::raw::c_void;

    fn execute(source_file: &str, func_name: &str) -> Result<(), CudaErrorWrapper> {
        // 1. 初始化 CUDA
        cuda::init()?;
        let device = cuda::get_device()?;

        // 2. 获取计算能力
        let (major_version, minor_version) = cuda::get_compute_capability(device)?;

        // 3. 编译 simple.ll 为 PTX
        let arch = format!("-arch=compute_{}{}", major_version, minor_version);
        let ptx = nvvm::compile_ll_to_ptx(source_file, &arch)?;

        // 4. 创建 CUDA 上下文
        let _context = cuda::create_context(device)?;
        // 5. 加载 PTX 模块
        let module = cuda::load_module(&ptx)?;
        // 6. 获取 kernel function
        let kernel = cuda::get_function(module, func_name)?;

        // 7. 分配参数并运行 kernel，取回结果
        let n = 16;
        let mut host_data = vec![0i32; n];
        let device_ptr = cuda::malloc(n * size_of::<i32>())?;
        cuda::memcpy_htod(device_ptr, host_data.as_ptr() as *const c_void, n * size_of::<i32>())?;
        let mut kernel_param = device_ptr as *mut c_void;
        let mut kernel_params = [(&mut kernel_param as *mut _) as *mut c_void];
        cuda::launch_kernel(kernel, (1, 1, 1), (n as u32, 1, 1), &mut kernel_params)?;
        cuda::memcpy_dtoh(host_data.as_mut_ptr() as *mut c_void, device_ptr, n * size_of::<i32>())?;
        println!("Kernel 计算结果: {:?}", host_data);
        cuda::free(device_ptr)?;
        Ok(())
    }

    #[test]
    fn test_01() -> Result<(), CudaErrorWrapper> {
        execute("example/simple.ll", "simple")
    }
}
