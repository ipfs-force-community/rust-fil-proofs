use std::{
    collections::HashMap,
    fs::OpenOptions,
    path::Path,
    sync::{Mutex, MutexGuard},
};

use log::{info, warn};
use mapr::{MmapMut, MmapOptions};

use crate::stacked::vanilla::numa::NumaNodeIndex;

// memory_size -> memory
type ShmPool = HashMap<usize, Vec<Mutex<MmapMut>>>;

pub(super) struct NumaPool {
    /// The index of the numa_groups vec is numa_node_index
    numa_groups: Vec<ShmPool>,
}

impl NumaPool {
    /// Create an empty NumaPoll
    pub fn empty() -> Self {
        Self {
            numa_groups: Vec::new(),
        }
    }

    /// Create NumaPool with the given numa_shm_files
    ///
    /// The index of the numa_shm_files vec is numa_node_index,
    /// and each item of numa_shm_files is the shm file paths corresponding to numa_node_index
    pub fn new(numa_shm_files: Vec<impl IntoIterator<Item = impl AsRef<Path>>>) -> Self {
        let numa_groups = numa_shm_files
            .into_iter()
            .map(Self::load_shm_files)
            .collect();
        Self { numa_groups }
    }

    fn load_shm_files(shm_files: impl IntoIterator<Item = impl AsRef<Path>>) -> ShmPool {
        let mut shm_pool: HashMap<usize, Vec<Mutex<MmapMut>>> = HashMap::new();

        for p in shm_files.into_iter() {
            let p = p.as_ref();
            let shm_file = match OpenOptions::new().read(true).write(true).open(p) {
                Ok(file) => file,
                Err(e) => {
                    warn!(
                        "open shm file: '{}', {:?}. ignore this shm file.",
                        p.display(),
                        e
                    );
                    continue;
                }
            };

            let file_size = match shm_file.metadata() {
                Ok(meta) => meta.len(),
                Err(e) => {
                    warn!(
                        "get the size of the '{}' file: {:?}. ignore this shm file.",
                        p.display(),
                        e,
                    );
                    continue;
                }
            };

            let mmap = match unsafe { MmapOptions::new().lock().map_mut(&shm_file) } {
                Ok(mmap) => mmap,
                Err(e) => {
                    // fallback to not locked if permissions are not available
                    warn!(
                        "lock mmap shm file '{}': {:?}. falling back",
                        p.display(),
                        e
                    );
                    match unsafe { MmapOptions::new().map_mut(&shm_file) } {
                        Ok(mmap) => mmap,
                        Err(e) => {
                            warn!(
                                "mmap shm file '{}': {:?}. ignore this shm file.",
                                p.display(),
                                e
                            );
                            continue;
                        }
                    }
                }
            };
            info!("loaded shm file: {}", p.display());
            let mmap = Mutex::new(mmap);
            shm_pool
                .entry(file_size as usize)
                .or_insert_with(Vec::new)
                .push(mmap);
        }
        shm_pool
    }

    /// Acquire the shm memory for the specified size
    ///
    /// Acquire returns the memory of the NUMA node where the caller thread is located.
    /// Make sure that the caller thread and the thread that using the memory returned
    /// by this function are in the same NUMA node and make sure the thread that using
    /// the returned memory will not be dispatched to other NUMA nodes, otherwise the
    /// performance of using returned memory will be very low
    pub fn acquire(&self, size: usize) -> Option<MutexGuard<'_, MmapMut>> {
        let numa_group = self
            .numa_groups
            .get(current_numa_node().unwrap_or_default().raw() as usize)?;
        for l in numa_group.get(&size)? {
            match l.try_lock() {
                Ok(m) => return Some(m),
                Err(_) => {}
            }
        }
        None
    }
}

#[cfg(not(test))]
fn current_numa_node() -> Option<NumaNodeIndex> {
    use crate::stacked::vanilla::numa;
    numa::current_numa_node()
}

#[cfg(test)]
fn current_numa_node() -> Option<NumaNodeIndex> {
    *tests::CUR_NUMA_NODE.lock().unwrap()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Mutex;

    use lazy_static::lazy_static;

    use crate::stacked::vanilla::numa::NumaNodeIndex;

    use super::NumaPool;

    lazy_static! {
        pub(super) static ref CUR_NUMA_NODE: Mutex<Option<NumaNodeIndex>> = Mutex::new(None);
    }

    /// set current numa node for testing
    fn set_current_numa_node(curr: Option<NumaNodeIndex>) {
        *CUR_NUMA_NODE.lock().unwrap() = curr;
    }

    #[test]
    fn test_numa_pool() {
        let temp_dir = tempfile::tempdir().expect("Failed to create tempdir");
        let temp_dir_path = temp_dir.as_ref();

        fn size_fn(numa_node_idx: usize) -> usize {
            (numa_node_idx + 1) * 10
        }

        let numa_shm_files: Vec<_> = (0..2)
            .map(|numa_node_idx| {
                (0..2).map(move |i| {
                    let path = temp_dir_path.join(format!("numa_{}_{}", numa_node_idx, i));

                    fs::write(&path, " ".repeat(size_fn(numa_node_idx)))
                        .expect("Failed to write data");
                    path
                })
            })
            .collect();
        let numa_pool = NumaPool::new(numa_shm_files);

        let mut mems = Vec::new();
        for numa_node_idx in 0..2 {
            let size = size_fn(numa_node_idx);
            let no_exist_size = size + 1;

            set_current_numa_node(Some(NumaNodeIndex::new(numa_node_idx as u32)));

            for _ in 0..2 {
                // Test for normal memory acquire
                let mem = numa_pool.acquire(size);
                assert!(mem.is_some());
                mems.push(mem);

                // Test to acquire the shared memory of non-existent memory size
                assert!(
                    numa_pool.acquire(no_exist_size).is_none(),
                    "acquire non-existent memory size should return None"
                );
            }
        }

        // Test when NumaPool is empty
        for numa_node_idx in 0..2 {
            let size = size_fn(numa_node_idx);
            set_current_numa_node(Some(NumaNodeIndex::new(numa_node_idx as u32)));

            for _ in 0..2 {
                assert!(
                    numa_pool.acquire(size).is_none(),
                    "acquire memory from empty NumaPool should return None"
                );
            }
        }

        drop(mems);

        for numa_node_idx in 0..2 {
            let size = size_fn(numa_node_idx);
            set_current_numa_node(Some(NumaNodeIndex::new(numa_node_idx as u32)));

            for _ in 0..2 {
                // Test for normal memory acquire
                let mem = numa_pool.acquire(size);
                assert!(mem.is_some());
            }
        }
    }
}
