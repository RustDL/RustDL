use std::ffi::{CStr, CString, c_uint};
use std::os::raw::{c_char, c_int, c_void};

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

type CUresult = c_int;
type NvvmProgram = *mut c_void;

#[derive(Debug, thiserror::Error)]
pub enum CudaErrorWrapper {
    #[error("CUDA错误: {0}")]
    CudaError(String),

    #[error("NVVM错误: {0}")]
    NvvmError(String),

    #[error("其他错误: {0}")]
    Other(String),
}

pub mod cuda {
    use super::*;

    unsafe extern "system" {
        pub fn cuInit(flags: c_int) -> CUresult;
        pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
        pub fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> CUresult;
        pub fn cuDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> CUresult;
        pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, device: c_int) -> CUresult;
        pub fn cuGetErrorString(error: CUresult, p_str: *mut *const c_char) -> CUresult;
        pub fn cuCtxCreate(pctx: *mut *mut c_void, flags: c_uint, dev: c_int) -> CUresult;
        pub fn cuModuleLoadDataEx(
            module: *mut *mut c_void,
            image: *const c_void,
            num_options: c_uint,
            options: *const c_uint,
            option_values: *const *const c_void,
        ) -> CUresult;
        pub fn cuModuleGetFunction(
            hfunc: *mut *mut c_void,
            module: *mut c_void,
            name: *const c_char,
        ) -> CUresult;
        pub fn cuMemAlloc(dptr: *mut u64, bytesize: usize) -> CUresult;
        pub fn cuMemcpyHtoD(dstDevice: u64, srcHost: *const c_void, ByteCount: usize) -> CUresult;
        pub fn cuMemcpyDtoH(dstHost: *mut c_void, srcDevice: u64, ByteCount: usize) -> CUresult;
        pub fn cuMemFree(dptr: u64) -> CUresult;
        pub fn cuLaunchKernel(
            f: *mut c_void,
            gridDimX: u32,
            gridDimY: u32,
            gridDimZ: u32,
            blockDimX: u32,
            blockDimY: u32,
            blockDimZ: u32,
            sharedMemBytes: u32,
            hStream: *mut c_void,
            kernelParams: *mut *mut c_void,
            extra: *mut *mut c_void,
        ) -> CUresult;
    }
    pub fn check_cuda_error(code: i32) -> Result<(), CudaErrorWrapper> {
        if code == 0 {
            return Ok(());
        }
        unsafe {
            let mut err_str: *const c_char = std::ptr::null();
            cuGetErrorString(code, &mut err_str);
            let msg = CStr::from_ptr(err_str).to_string_lossy().into_owned();
            Err(CudaErrorWrapper::CudaError(msg))
        }
    }
    pub fn init() -> Result<(), CudaErrorWrapper> {
        check_cuda_error(unsafe { cuInit(0) })
    }
    pub fn get_device() -> Result<i32, CudaErrorWrapper> {
        let mut device = 0;
        check_cuda_error(unsafe { cuDeviceGet(&mut device, 0) })?;
        Ok(device)
    }
    pub fn get_device_name(device: i32) -> Result<String, CudaErrorWrapper> {
        let mut device_name = [0 as c_char; 256];
        check_cuda_error(unsafe { cuDeviceGetName(device_name.as_mut_ptr(), 256, device) })?;
        Ok(unsafe { CStr::from_ptr(device_name.as_ptr()) }
            .to_string_lossy()
            .to_string())
    }
    pub fn create_context(device: i32) -> Result<*mut c_void, CudaErrorWrapper> {
        let mut ctx = std::ptr::null_mut();
        check_cuda_error(unsafe { cuCtxCreate(&mut ctx, 0, device) })?;
        Ok(ctx)
    }
    pub fn load_module(ptx: &[u8]) -> Result<*mut c_void, CudaErrorWrapper> {
        let mut module = std::ptr::null_mut();
        check_cuda_error(unsafe {
            cuModuleLoadDataEx(
                &mut module,
                ptx.as_ptr() as *const c_void,
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        })?;
        Ok(module)
    }
    pub fn get_function(module: *mut c_void, name: &str) -> Result<*mut c_void, CudaErrorWrapper> {
        let mut func = std::ptr::null_mut();
        let cname = CString::new(name).unwrap();
        check_cuda_error(unsafe { cuModuleGetFunction(&mut func, module, cname.as_ptr()) })?;
        Ok(func)
    }
    pub fn malloc(size: usize) -> Result<u64, CudaErrorWrapper> {
        let mut ptr = 0u64;
        check_cuda_error(unsafe { cuMemAlloc(&mut ptr, size) })?;
        Ok(ptr)
    }
    pub fn memcpy_htod(dst: u64, src: *const c_void, size: usize) -> Result<(), CudaErrorWrapper> {
        check_cuda_error(unsafe { cuMemcpyHtoD(dst, src, size) })
    }
    pub fn memcpy_dtoh(dst: *mut c_void, src: u64, size: usize) -> Result<(), CudaErrorWrapper> {
        check_cuda_error(unsafe { cuMemcpyDtoH(dst, src, size) })
    }
    pub fn free(ptr: u64) -> Result<(), CudaErrorWrapper> {
        check_cuda_error(unsafe { cuMemFree(ptr) })
    }
    pub fn launch_kernel(
        func: *mut c_void,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &mut [*mut c_void],
    ) -> Result<(), CudaErrorWrapper> {
        check_cuda_error(unsafe {
            cuLaunchKernel(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }
    pub fn get_compute_capability(device: i32) -> Result<(i32, i32), CudaErrorWrapper> {
        let mut major = 0;
        let mut minor = 0;
        check_cuda_error(unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        })?;
        check_cuda_error(unsafe {
            cuDeviceGetAttribute(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            )
        })?;
        Ok((major, minor))
    }
}

pub mod nvvm {
    use super::*;

    unsafe extern "system" {
        pub fn nvvmCreateProgram(prog: *mut NvvmProgram) -> i32;
        pub fn nvvmAddModuleToProgram(
            prog: NvvmProgram,
            buffer: *const c_char,
            size: usize,
            name: *const c_char,
        ) -> i32;
        pub fn nvvmCompileProgram(
            prog: NvvmProgram,
            numOptions: c_int,
            options: *const *const c_char,
        ) -> i32;
        pub fn nvvmGetCompiledResultSize(prog: NvvmProgram, bufferSizeRet: *mut usize) -> i32;
        pub fn nvvmGetCompiledResult(prog: NvvmProgram, buffer: *mut c_char) -> i32;
        pub fn nvvmDestroyProgram(prog: *mut NvvmProgram) -> i32;
        pub fn nvvmGetErrorString(result: i32) -> *const c_char;
    }
    pub fn check_nvvm_error(code: i32) -> Result<(), CudaErrorWrapper> {
        if code == 0 {
            return Ok(());
        }
        unsafe {
            let err_str = nvvmGetErrorString(code);
            let msg = CStr::from_ptr(err_str).to_string_lossy().into_owned();
            Err(CudaErrorWrapper::NvvmError(msg))
        }
    }
    pub fn compile_ll_to_ptx(filename: &str, arch: &str) -> Result<Vec<u8>, CudaErrorWrapper> {
        let source = std::fs::read_to_string(filename)
            .map_err(|e| CudaErrorWrapper::Other(e.to_string()))?;
        let filename = CString::new(filename).unwrap();
        let source = CString::new(source).unwrap();
        let mut program = std::ptr::null_mut();
        check_nvvm_error(unsafe { nvvmCreateProgram(&mut program) })?;
        check_nvvm_error(unsafe {
            nvvmAddModuleToProgram(
                program,
                source.as_ptr(),
                source.count_bytes(),
                filename.as_ptr(),
            )
        })?;
        let arch_cstr = CString::new(arch).unwrap();
        let options = [arch_cstr.as_ptr()];
        check_nvvm_error(unsafe { nvvmCompileProgram(program, 1, options.as_ptr()) })?;
        let mut ptx_size: usize = 0;
        check_nvvm_error(unsafe { nvvmGetCompiledResultSize(program, &mut ptx_size) })?;
        let mut buffer = vec![0u8; ptx_size];
        check_nvvm_error(unsafe {
            nvvmGetCompiledResult(program, buffer.as_mut_ptr() as *mut c_char)
        })?;
        check_nvvm_error(unsafe { nvvmDestroyProgram(&mut program) })?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod test_cuda {
    use super::*;

    #[test]
    fn it_works() -> Result<(), CudaErrorWrapper> {
        // 1. 初始化 CUDA
        cuda::init()?;
        let device = cuda::get_device()?;
        let device_name = cuda::get_device_name(device)?;
        println!("使用设备: {}", device_name);

        // 2. 获取计算能力
        let (major_version, minor_version) = cuda::get_compute_capability(device)?;
        println!("计算能力: {}.{}", major_version, minor_version);

        // 3. 编译 example.ll 为 PTX
        let arch = format!("-arch=compute_{}{}", major_version, minor_version);
        let ptx = nvvm::compile_ll_to_ptx("example.ll", &arch)?;

        // 4. 创建 CUDA 上下文
        let _context = cuda::create_context(device)?;
        // 5. 加载 PTX 模块
        let module = cuda::load_module(&ptx)?;
        // 6. 获取 kernel function
        let kernel = cuda::get_function(module, "simple")?;

        // 7. 分配参数并运行 kernel，取回结果
        let n = 16;
        let mut host_data = vec![0i32; n];
        let device_ptr = cuda::malloc(n * std::mem::size_of::<i32>())?;
        cuda::memcpy_htod(
            device_ptr,
            host_data.as_ptr() as *const c_void,
            n * std::mem::size_of::<i32>(),
        )?;
        let mut kernel_param = device_ptr as *mut c_void;
        let mut kernel_params = [&mut kernel_param as *mut _ as *mut c_void];
        cuda::launch_kernel(kernel, (1, 1, 1), (n as u32, 1, 1), &mut kernel_params)?;
        cuda::memcpy_dtoh(
            host_data.as_mut_ptr() as *mut c_void,
            device_ptr,
            n * std::mem::size_of::<i32>(),
        )?;
        println!("Kernel 计算结果: {:?}", host_data);
        cuda::free(device_ptr)?;
        Ok(())
    }
}

#[cfg(test)]
mod test_cuda_macro {
    use rust_cuda_compiler::cuda;

    #[cuda]
    fn test_cuda_macro(a: i32, b: i32) -> i32 {
        a + b
    }

    #[test]
    fn test_macro() {
        let result = test_cuda_macro(2, 3);
        assert_eq!(result, 5);
    }
}