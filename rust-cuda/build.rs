use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda_lite.c");

    let cuda_info = detect_cuda().expect("Failed to detect CUDA installation");
    println!("cargo:warning=Found CUDA at: {}", cuda_info.root.display());
    println!("cargo:warning=CUDA version: {}", cuda_info.version);

    for lib_path in &cuda_info.lib_paths {
        println!("cargo:warning=lib path: {}", lib_path.display());
        println!("cargo:rustc-link-search=native={}", lib_path.display());
        println!("cargo:rustc-env=PATH={};{}", env::var("PATH").unwrap_or_default(), lib_path.display());
    }

    println!("cargo:warning=include path: {}", cuda_info.include_path.display());
    println!("cargo:include={}", cuda_info.include_path.display());

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvvm");
}

struct CudaInfo {
    root: PathBuf,
    version: String,
    lib_paths: Vec<PathBuf>,
    include_path: PathBuf,
}

fn detect_cuda() -> Result<CudaInfo, Box<dyn std::error::Error>> {
    if let Some(nvcc_path) = find_nvcc() {
        return parse_nvcc_path(nvcc_path);
    }

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        return parse_cuda_home(PathBuf::from(cuda_home));
    }

    let default_paths = get_default_cuda_paths();
    for path in default_paths {
        if path.exists() {
            return parse_cuda_home(path);
        }
    }

    Err(
        "CUDA installation not found. Please install CUDA or set CUDA_HOME environment variable."
            .into(),
    )
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(path) = env::var("PATH") {
        for dir in path.split(";") {
            let nvcc_path = Path::new(dir).join("nvcc");
            let nvcc_path_exe = Path::new(dir).join("nvcc.exe");

            if nvcc_path.exists() {
                return Some(nvcc_path);
            } else if nvcc_path_exe.exists() {
                return Some(nvcc_path_exe);
            }
        }
    }

    if cfg!(target_os = "windows") {
        if let Ok(output) = Command::new("where")
            .arg("nvcc")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        {
            if !output.is_empty() {
                return Some(PathBuf::from(output.split('\n').next().unwrap()));
            }
        }
    }

    if cfg!(unix) {
        if let Ok(output) = Command::new("which")
            .arg("nvcc")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        {
            if !output.is_empty() {
                return Some(PathBuf::from(output));
            }
        }
    }

    None
}

fn parse_nvcc_path(nvcc_path: PathBuf) -> Result<CudaInfo, Box<dyn std::error::Error>> {
    let nvcc_dir = nvcc_path.parent().ok_or("Invalid nvcc path")?;
    let cuda_root = nvcc_dir.parent().ok_or("Invalid CUDA root directory")?;
    let version = get_cuda_version(&nvcc_path)?;

    let mut lib_paths = Vec::new();
    if cfg!(target_os = "windows") {
        lib_paths.push(cuda_root.join("lib").join("x64"));
        lib_paths.push(cuda_root.join("bin"));

        lib_paths.push(cuda_root.join("nvvm").join("lib").join("x64"));
        lib_paths.push(cuda_root.join("nvvm").join("bin"));
    } else if cfg!(target_os = "macos") {
        lib_paths.push(cuda_root.join("lib"));
    } else {
        // Linux
        lib_paths.push(cuda_root.join("lib64"));
        lib_paths.push(cuda_root.join("lib"));
    }

    Ok(CudaInfo {
        root: cuda_root.to_path_buf(),
        version,
        lib_paths,
        include_path: cuda_root.join("include"),
    })
}

fn parse_cuda_home(cuda_home: PathBuf) -> Result<CudaInfo, Box<dyn std::error::Error>> {
    if !cuda_home.exists() {
        return Err(format!("CUDA_HOME path does not exist: {}", cuda_home.display()).into());
    }

    let version = if let Some(nvcc_path) = find_nvcc_in_path(&cuda_home) {
        get_cuda_version(&nvcc_path)?
    } else {
        "unknown".to_string()
    };

    let mut lib_paths = Vec::new();

    if cfg!(target_os = "windows") {
        lib_paths.push(cuda_home.join("lib").join("x64"));
        lib_paths.push(cuda_home.join("bin"));

        lib_paths.push(cuda_home.join("nvvm").join("lib").join("x64"));
        lib_paths.push(cuda_home.join("nvvm").join("bin"));
    } else if cfg!(target_os = "macos") {
        lib_paths.push(cuda_home.join("lib"));
    } else {
        // Linux
        lib_paths.push(cuda_home.join("lib64"));
        lib_paths.push(cuda_home.join("lib"));
    }

    Ok(CudaInfo {
        root: cuda_home.clone(),
        version,
        lib_paths,
        include_path: cuda_home.join("include"),
    })
}

fn find_nvcc_in_path(base_path: &Path) -> Option<PathBuf> {
    let bin_dir = base_path.join("bin");
    let nvcc_path = bin_dir.join("nvcc");
    let nvcc_path_exe = bin_dir.join("nvcc.exe");

    if nvcc_path.exists() {
        Some(nvcc_path)
    } else if nvcc_path_exe.exists() {
        Some(nvcc_path_exe)
    } else {
        None
    }
}

fn get_cuda_version(nvcc_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new(nvcc_path)
        .arg("--version")
        .output()
        .expect("Failed to execute nvcc");

    let version_text = String::from_utf8_lossy(&output.stdout).to_string();

    // 从输出中提取版本号
    if let Some(cuda_release) = version_text
        .lines()
        .find(|line| line.contains("Cuda compilation tools"))
    {
        if let Some(version) = cuda_release.split(',').nth(1) {
            return Ok(version.trim().to_string());
        }
    }

    Err("Failed to parse CUDA version".into())
}

fn get_default_cuda_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if cfg!(target_os = "windows") {
        paths.push(PathBuf::from(
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        ));
        paths.push(PathBuf::from("C:\\CUDA"));
    } else if cfg!(target_os = "macos") {
        paths.push(PathBuf::from("/Developer/NVIDIA/CUDA-"));
        paths.push(PathBuf::from("/usr/local/cuda"));
    } else {
        paths.push(PathBuf::from("/usr/local/cuda"));
        paths.push(PathBuf::from("/opt/cuda"));
    }

    paths
}
