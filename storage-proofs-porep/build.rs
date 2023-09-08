use std::{env, path::PathBuf};

fn main() {

    let ipfs_fpga_dir_neptune =
        PathBuf::from(env::var("FAAS_IPFS_PATH").expect("FAAS_IPFS_PATH env var is not defined"));
    let neptune_dir = dunce::canonicalize(ipfs_fpga_dir_neptune.join("neptune_plus")).unwrap();
    let poseidon_dir = dunce::canonicalize(ipfs_fpga_dir_neptune.join("libposeidon")).unwrap();
    println!(
        "cargo:rustc-link-search=native={}",
        env::join_paths(&[neptune_dir]).unwrap().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=dylib=neptune_plus_ffi");
    println!(
        "cargo:rustc-link-search=native={}",
        env::join_paths(&[poseidon_dir]).unwrap().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=dylib=poseidon_hash");
    println!("cargo:rustc-link-search=native=/opt/xilinx/xrt/lib");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=xilinxopencl");
    println!("cargo:rustc-link-lib=dylib=xrt_core");
}

