
use std::error::Error;
use std::ffi::{c_char, CStr};

use libloading::Library;

type CuInit                = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet           = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuDeviceGetName       = unsafe extern "C" fn(*mut c_char, i32, i32) -> i32;
type CuDeviceTotalMem      = unsafe extern "C" fn(*mut usize, i32) -> i32;
type CuDeviceGetAttribute  = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;

const MAX_THREADS_BLOCK    : i32 = 1;
const BLOCK_DIM_X          : i32 = 2;
const BLOCK_DIM_Y          : i32 = 3;
const BLOCK_DIM_Z          : i32 = 4;
const GRID_DIM_X           : i32 = 5;
const GRID_DIM_Y           : i32 = 6;
const GRID_DIM_Z           : i32 = 7;
const SHARED_MEMORY_BLOCK  : i32 = 8;
const WARP_SIZE            : i32 = 10;
const SM_COUNT             : i32 = 16;
const L2_CACHE_SIZE        : i32 = 38;
const MAX_THREADS_SM       : i32 = 39;
const COMPUTE_MAJOR        : i32 = 75;
const COMPUTE_MINOR        : i32 = 76;
const SHARED_MEMORY_SM     : i32 = 81;
const REGISTERS_SM         : i32 = 82;

fn check(status: i32, operation: &str) -> Result<(), Box<dyn Error>> {
    if status != 0 {
        return Err(format!("{operation}: erro CUDA {status}").into());
    }

    Ok(())
}

fn attribute(
    get_attribute: CuDeviceGetAttribute,
    device: i32,
    id: i32,
) -> Result<i32, Box<dyn Error>> {
    let mut value = 0;

    check(
        unsafe { get_attribute(&mut value, id, device) },
        "cuDeviceGetAttribute",
    )?;

    Ok(value)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cuda = unsafe { Library::new("libcuda.so.1")? };

    let init: CuInit = unsafe {
        *cuda.get(b"cuInit\0")?
    };

    let device_get: CuDeviceGet = unsafe {
        *cuda.get(b"cuDeviceGet\0")?
    };

    let get_name: CuDeviceGetName = unsafe {
        *cuda.get(b"cuDeviceGetName\0")?
    };

    let total_mem: CuDeviceTotalMem = unsafe {
        *cuda.get(b"cuDeviceTotalMem_v2\0")?
    };

    let get_attribute: CuDeviceGetAttribute = unsafe {
        *cuda.get(b"cuDeviceGetAttribute\0")?
    };

    check(unsafe { init(0) }, "cuInit")?;

    let mut device = 0;

    check(
        unsafe { device_get(&mut device, 0) },
        "cuDeviceGet",
    )?;

    let mut name = [0 as c_char; 256];

    check(
        unsafe {
            get_name(name.as_mut_ptr(), name.len() as i32, device)
        },
        "cuDeviceGetName",
    )?;

    let name = unsafe {
        CStr::from_ptr(name.as_ptr())
    }.to_string_lossy();

    let get = |id| attribute(get_attribute, device, id);

    let major         = get(COMPUTE_MAJOR)?;
    let minor         = get(COMPUTE_MINOR)?;

    let sms           = get(SM_COUNT)?;
    let warp          = get(WARP_SIZE)?;
    let threads_block = get(MAX_THREADS_BLOCK)?;
    let threads_sm    = get(MAX_THREADS_SM)?;

    let block_x       = get(BLOCK_DIM_X)?;
    let block_y       = get(BLOCK_DIM_Y)?;
    let block_z       = get(BLOCK_DIM_Z)?;

    let grid_x        = get(GRID_DIM_X)?;
    let grid_y        = get(GRID_DIM_Y)?;
    let grid_z        = get(GRID_DIM_Z)?;

    let shared_block  = get(SHARED_MEMORY_BLOCK)?;
    let shared_sm     = get(SHARED_MEMORY_SM)?;
    let registers_sm  = get(REGISTERS_SM)?;
    let l2_cache      = get(L2_CACHE_SIZE)?;

    let mut vram: usize = 0;

    check(
        unsafe { total_mem(&mut vram, device) },
        "cuDeviceTotalMem",
    )?;

    println!("GPU:                 {name}");
    println!("Compute capability:  {major}.{minor}");

    println!("\nExecucao:");
    println!("SMs:                 {sms}");
    println!("Warp size:           {warp}");
    println!("Threads / block:     {threads_block}");
    println!("Threads / SM:        {threads_sm}");
    println!("Threads / block xyz: {block_x} {block_y} {block_z}");
    println!("Grid xyz:            {grid_x} {grid_y} {grid_z}");

    println!("\nMemoria:");
    println!("VRAM:                {:.2} GiB",
             vram as f64 / 1024_f64.powi(3));
    println!("Shared / block:      {shared_block} bytes");
    println!("Shared / SM:         {shared_sm} bytes");
    println!("Registers / SM:      {registers_sm}");
    println!("L2 cache:            {l2_cache} bytes");

    Ok(())
}
