//! With the `cuda` feature on MSVC, the CUDA runtime, cuBLAS and the driver are
//! delay-loaded: they cost ~140 MB of resident memory, and only the embedding model
//! needs them, so commands that never embed (benchmarks on cached vectors) skip them.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    if cuda && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        for dll in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll", "nvcuda.dll"] {
            println!("cargo:rustc-link-arg-bins=/DELAYLOAD:{dll}");
        }
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
        println!("cargo:rustc-link-arg-bins=/IGNORE:4199"); // a DLL this CUDA version does not import
    }
}
