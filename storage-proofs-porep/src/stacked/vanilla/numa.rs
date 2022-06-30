use libc::{self, c_int};

#[link(name = "numa", kind = "dylib")]
extern "C" {
    /// Check if NUMA support is enabled. Returns -1 if not enabled, in which case other functions will undefined
    fn numa_available() -> c_int;
    ///  Returns the NUMA node corresponding to a CPU, or -1 if the CPU is invalid
    fn numa_node_of_cpu(cpu: c_int) -> c_int;
}

lazy_static! {
    static ref NUMA_AVAILABLE: bool = unsafe { numa_available() >= 0 };
}

/// Index of a NUMA node.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct NumaNodeIndex(u32);

impl NumaNodeIndex(u32) {
    /// Returns NUMA node index of c_int type
    pub fn raw(&self) -> c_int {
        self.0 as c_int
    }
}

/// Returns the current NUMA node on which the thread is running.
///
/// Since threads may migrate to another node with the scheduler,
/// you need to bind the current worker thread to the specified core when calling this function
pub fn current_node() -> NumaNodeIndex {
    unsafe {
        if !*NUMA_AVAILABLE {
            return NumaNodeIndex(0);
        }
        let cpu = libc::sched_getcpu();
        // Use 0 as fallback if sched_getcpu call fails
        if cpu < 0 {
            return NumaNodeIndex(0);
        }
        let node = unsafe { numa_node_of_cpu(cpu) };
        // If libnuma cannot find the appropriate node, then use 0 as fallback.
        if node < 0 {
            return NumaNodeIndex(0);
        }
        NumaNodeIndex(node as u32)
    }
}
