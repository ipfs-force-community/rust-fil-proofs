use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fs::File;
use std::hint::spin_loop;
use std::marker::{PhantomData, Sync};
use std::mem::{size_of, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    MutexGuard, Once,
};

use anyhow::{Context, Result};
use byte_slice_cast::{AsSliceOf, FromByteSlice};
use log::{debug, info, warn};
use mapr::{Mmap, MmapMut, MmapOptions};
use storage_proofs_core::settings::{ShmNumaDirPattern, SETTINGS};

mod shm;

pub struct CacheReader<T> {
    file: File,
    bufs: UnsafeCell<[Mmap; 2]>,
    size: usize,
    degree: usize,
    window_size: usize,
    cursor: IncrementingCursor,
    consumer: AtomicU64,
    _t: PhantomData<T>,
}

unsafe impl<T> Sync for CacheReader<T> {}

struct IncrementingCursor {
    cur: AtomicUsize,
    cur_safe: AtomicUsize,
}

fn compare_and_swap(atomic: &AtomicUsize, before: usize, after: usize) -> usize {
    match atomic.compare_exchange_weak(before, after, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(x) => {
            assert_eq!(x, before);
            before
        }
        _ => after,
    }
}

/// IncrementingCursor provides an atomic variable which can be incremented such that only one thread attempting the
/// increment is selected to perform actions required to effect the transition. Unselected threads wait until the
/// transition has completed. Transition and wait condition are both specified by closures supplied by the caller.
impl IncrementingCursor {
    fn new(val: usize) -> Self {
        Self {
            cur: AtomicUsize::new(val),
            cur_safe: AtomicUsize::new(val),
        }
    }

    fn store(&self, val: usize) {
        self.cur.store(val, Ordering::SeqCst);
        self.cur_safe.store(val, Ordering::SeqCst);
    }

    fn compare_and_swap(&self, before: usize, after: usize) {
        compare_and_swap(&self.cur, before, after);
        compare_and_swap(&self.cur_safe, before, after);
    }

    fn increment<F: Fn() -> bool, G: Fn()>(&self, target: usize, wait_fn: F, advance_fn: G) {
        // Check using `cur_safe`, to ensure we wait until the current cursor value is safe to use.
        // If we were to instead check `cur`, it could have been incremented but not yet safe.
        let cur = self.cur_safe.load(Ordering::SeqCst);
        if target > cur {
            // Only one producer will successfully increment `cur`. We need this second atomic because we cannot
            // increment `cur_safe` until after the underlying resource has been advanced.
            let instant_cur = compare_and_swap(&self.cur, cur, cur + 1);
            if instant_cur == cur {
                // We successfully incremented `self.cur`, so we are responsible for advancing the resource.
                {
                    while wait_fn() {
                        spin_loop()
                    }
                }

                advance_fn();

                // Now it is safe to use the new window.
                self.cur_safe.fetch_add(1, Ordering::SeqCst);
            } else {
                // We failed to increment `self.cur_window`, so we must wait for the window to be advanced before
                // continuing. Wait until it is safe to use the new current window.
                while self.cur_safe.load(Ordering::SeqCst) != cur + 1 {
                    spin_loop()
                }
            }
        }
    }
}

impl<T: FromByteSlice> CacheReader<T> {
    pub fn new(filename: &Path, window_size: Option<usize>, degree: usize) -> Result<Self> {
        info!("initializing cache");
        let file = File::open(filename)?;
        let size = File::metadata(&file)?.len() as usize;
        let window_size = match window_size {
            Some(s) => {
                if s < size {
                    assert_eq!(
                        0,
                        size % degree * size_of::<T>(),
                        "window size is not multiple of element size"
                    );
                };
                s
            }
            None => {
                let num_windows = 8;
                assert_eq!(0, size % num_windows);
                size / num_windows
            }
        };

        let buf0 = Self::map_buf(0, window_size, &file)?;
        let buf1 = Self::map_buf(window_size as u64, window_size, &file)?;
        Ok(Self {
            file,
            bufs: UnsafeCell::new([buf0, buf1]),
            size,
            degree,
            window_size,
            // The furthest window from which the cache has yet been read.
            cursor: IncrementingCursor::new(0),
            consumer: AtomicU64::new(0),
            _t: PhantomData::<T>,
        })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn window_nodes(&self) -> usize {
        self.size() / (size_of::<T>() * self.degree)
    }

    /// Safety: incrementing the consumer at the end of a window will unblock the producer waiting to remap the
    /// consumer's previous buffer. The buffer must not be accessed once this has happened.
    pub unsafe fn increment_consumer(&self) {
        self.consumer.fetch_add(1, Ordering::SeqCst);
    }

    pub fn store_consumer(&self, val: u64) {
        self.consumer.store(val, Ordering::SeqCst);
    }

    pub fn get_consumer(&self) -> u64 {
        self.consumer.load(Ordering::SeqCst)
    }

    #[inline]
    fn get_bufs(&self) -> &[Mmap] {
        unsafe { std::slice::from_raw_parts((*self.bufs.get()).as_ptr(), 2) }
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut_bufs(&self) -> &mut [Mmap] {
        slice::from_raw_parts_mut((*self.bufs.get()).as_mut_ptr(), 2)
    }

    #[allow(dead_code)]
    // This is unused, but included to document the meaning of its components.
    // This allows splitting the reset in order to avoid a pause.
    pub fn reset(&self) -> Result<()> {
        self.start_reset()?;
        self.finish_reset()
    }

    pub fn start_reset(&self) -> Result<()> {
        let buf0 = Self::map_buf(0, self.window_size, &self.file)?;
        let bufs = unsafe { self.get_mut_bufs() };
        bufs[0] = buf0;
        Ok(())
    }

    pub fn finish_reset(&self) -> Result<()> {
        let buf1 = Self::map_buf(self.window_size as u64, self.window_size, &self.file)?;
        let bufs = unsafe { self.get_mut_bufs() };
        bufs[1] = buf1;
        self.cursor.store(0);
        self.store_consumer(0);
        Ok(())
    }

    fn map_buf(offset: u64, len: usize, file: &File) -> Result<Mmap> {
        unsafe {
            MmapOptions::new()
                .offset(offset)
                .len(len)
                .private()
                .map(file)
                .map_err(|e| e.into())
        }
    }

    #[inline]
    fn window_element_count(&self) -> usize {
        self.window_size / size_of::<T>()
    }

    /// `pos` is in units of `T`.
    #[inline]
    /// Safety: A returned slice must not be accessed once the buffer from which it has been derived is remapped. A
    /// buffer will never be remapped until the `consumer` atomic contained in `self` has been advanced past the end of
    /// the window. NOTE: each time `consumer` is incremented, `self.degrees` elements of the cache are invalidated.
    /// This means callers should only access slice elements sequentially. They should only call `increment_consumer`
    /// once the next `self.degree` elements of the cache will never be accessed again.
    pub unsafe fn consumer_slice_at(&self, pos: usize) -> &[T] {
        assert!(
            pos < self.size,
            "pos {} out of range for buffer of size {}",
            pos,
            self.size
        );
        let window = pos / self.window_element_count();
        let pos = pos % self.window_element_count();
        let targeted_buf = &self.get_bufs()[window % 2];

        &targeted_buf.as_slice_of::<T>().expect("as_slice_of failed")[pos..]
    }

    /// `pos` is in units of `T`.
    #[inline]
    /// Safety: This call may advance the rear buffer, making it unsafe to access slices derived from that buffer again.
    /// It is the callers responsibility to ensure such illegal access is not attempted. This can be prevented if users
    /// never access values past which the cache's `consumer` atomic has been incremented. NOTE: each time `consumer` is
    /// incremented, `self.degrees` elements of the cache are invalidated.
    pub unsafe fn slice_at(&self, pos: usize) -> &[T] {
        assert!(
            pos < self.size,
            "pos {} out of range for buffer of size {}",
            pos,
            self.size
        );
        let window = pos / self.window_element_count();
        if window == 1 {
            self.cursor.compare_and_swap(0, 1);
        }

        let pos = pos % self.window_element_count();

        let wait_fn = || {
            let safe_consumer = (window - 1) * (self.window_element_count() / self.degree);
            (self.consumer.load(Ordering::SeqCst) as usize) < safe_consumer
        };

        self.cursor
            .increment(window, &wait_fn, &|| self.advance_rear_window(window));

        let targeted_buf = &self.get_bufs()[window % 2];

        &targeted_buf.as_slice_of::<T>().expect("as_slice_of failed")[pos..]
    }

    fn advance_rear_window(&self, new_window: usize) {
        assert!(new_window as usize * self.window_size < self.size);

        let replace_idx = (new_window % 2) as usize;

        let new_buf = Self::map_buf(
            (new_window * self.window_size) as u64,
            self.window_size as usize,
            &self.file,
        )
        .expect("map_buf failed");

        unsafe {
            self.get_mut_bufs()[replace_idx] = new_buf;
        }
    }
}

fn normal_allocate_layer(sector_size: usize) -> Result<MmapMut> {
    match MmapOptions::new()
        .len(sector_size)
        .private()
        .clone()
        .lock()
        .map_anon()
        .and_then(|mut layer| {
            layer.mlock()?;
            Ok(layer)
        }) {
        Ok(layer) => Ok(layer),
        Err(err) => {
            // fallback to not locked if permissions are not available
            warn!("failed to lock map {:?}, falling back", err);
            let layer = MmapOptions::new().len(sector_size).private().map_anon()?;
            Ok(layer)
        }
    }
}

#[derive(Debug)]
pub enum Memory {
    Normal(MmapMut),
    Shm(MutexGuard<'static, MmapMut>),
}

impl Deref for Memory {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Memory::Normal(m) => m.deref(),
            Memory::Shm(m) => m.deref(),
        }
    }
}

impl DerefMut for Memory {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Memory::Normal(m) => m.deref_mut(),
            Memory::Shm(m) => m.deref_mut(),
        }
    }
}

impl AsRef<[u8]> for Memory {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

impl AsMut<[u8]> for Memory {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.deref_mut()
    }
}

fn allocate_layer(sector_size: usize) -> Result<Memory> {
    Ok(match numa_pool().acquire(sector_size) {
        Some(shm_mem) => Memory::Shm(shm_mem),
        None => {
            debug!("unable to load shm memory, falling back.");
            Memory::Normal(normal_allocate_layer(sector_size)?)
        }
    })
}

pub fn setup_create_label_memory(
    sector_size: usize,
    degree: usize,
    window_size: Option<usize>,
    cache_path: &Path,
) -> Result<(CacheReader<u32>, Memory, Memory)> {
    let parents_cache = CacheReader::new(cache_path, window_size, degree)?;
    let layer_labels = allocate_layer(sector_size)?;
    let exp_labels = allocate_layer(sector_size)?;

    Ok((parents_cache, layer_labels, exp_labels))
}

/// Get the static global NUMA pool reference
fn numa_pool() -> &'static shm::NumaPool {
    fn init_numa_pool() -> Result<shm::NumaPool> {
        let shm_numa_dir_pattern = SETTINGS.multicore_sdr_shm_numa_dir_pattern("/dev/shm");

        Ok(shm::NumaPool::new(scan_shm_files(&shm_numa_dir_pattern)?))
    }

    static mut NUMA_POOL: MaybeUninit<shm::NumaPool> = MaybeUninit::uninit();
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let numa_pool = match init_numa_pool() {
            Ok(p) => p,
            Err(e) => {
                warn!("init numa pool: {:?}", e);
                shm::NumaPool::empty()
            }
        };

        unsafe { NUMA_POOL = MaybeUninit::new(numa_pool) }
    });

    unsafe { NUMA_POOL.assume_init_ref() }
}

/// Scan SHM files by the given `shm_numa_dir_pattern`
///
/// let p = ShmNumaDirPattern::new("/dev/shm/filecoin-proof-label/numa_$NUMA_NODE_INDEX", "/dev/shm");
/// scan_shm_files(&p);
///
/// $NUMA_NODE_INDEX corresponds to the index of the numa node,
/// and you should make sure that the shared memory files stored in this folder were created on that numa node ($NUMA_NODE_INDEX)
/// In the above example, the following shared memory files will be matched.
///
/// NUMA node 0:
/// /dev/shm/filecoin-proof-label/numa_0/mem_32GiB_1
/// /dev/shm/filecoin-proof-label/numa_0/mem_64GiB_1
/// /dev/shm/filecoin-proof-label/numa_0/any_file_name
/// NUMA node 1:
/// /dev/shm/filecoin-proof-label/numa_1/mem_32GiB_1
/// /dev/shm/filecoin-proof-label/numa_1/mem_64GiB_1
/// /dev/shm/filecoin-proof-label/numa_1/any_file_name
/// NUMA node N:
/// ...
fn scan_shm_files(shm_numa_dir_pattern: &ShmNumaDirPattern) -> Result<Vec<Vec<PathBuf>>> {
    use glob::glob;
    use regex::Regex;

    let re_numa_node_idx = Regex::new(shm_numa_dir_pattern.to_regex_pattern().as_str())
        .context("invalid `multicore_sdr_shm_numa_dir_pattern`")?;

    // numa_shm_files_map: { NUMA_NODE_INDEX -> Vec<PathBuf of this numa node shm file> }
    let mut numa_shm_files_map = HashMap::new();
    glob(shm_numa_dir_pattern.to_glob_pattern().as_str())
        .context("invalid `multicore_sdr_shm_numa_dir_pattern`")?
        .filter_map(|path_res| path_res.ok())
        .filter_map(|path| {
            let numa_node_idx: usize = re_numa_node_idx
                .captures(path.to_str()?)?
                .get(1)?
                .as_str()
                .parse()
                .ok()?;
            Some((numa_node_idx, path))
        })
        .for_each(|(numa_node_idx, path)| {
            numa_shm_files_map
                .entry(numa_node_idx)
                .or_insert_with(|| Vec::new())
                .push(path);
        });

    // Converts the numa_shm_files_map { NUMA_NODE_INDEX -> Vec<PathBuf of this numa node shm file> }
    // to numa_shm_files Vec [ NUMA_NODE_INDEX -> Vec<PathBuf of this numa node shm file> ]
    let numa_shm_files = match numa_shm_files_map.keys().max() {
        Some(&max_node_idx) => {
            let mut numa_vec = Vec::with_capacity(max_node_idx + 1);
            for i in 0..=max_node_idx {
                numa_vec.push(numa_shm_files_map.remove(&i).unwrap_or_else(|| Vec::new()))
            }
            numa_vec
        }
        None => Vec::new(),
    };
    Ok(numa_shm_files)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        iter::repeat,
        path::{Path, PathBuf},
    };

    use rand::{distributions::Alphanumeric, prelude::ThreadRng, Rng};
    use storage_proofs_core::settings::ShmNumaDirPattern;

    use crate::stacked::vanilla::memory_handling::scan_shm_files;

    #[test]
    fn test_scan_shm_files() {
        const NUMA_NODE_IDX_VAR_NAME: &'static str = "$NUMA_NODE_INDEX";

        struct TestCase {
            shm_numa_dir_pattern: String,
            // { numa_node_idx -> shm files count of this numa node }
            numa_node_files: HashMap<usize, usize>,
        }
        let cases = vec![
            TestCase {
                shm_numa_dir_pattern: format!("abc/numa_{}", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 4), (1, 0), (3, 2)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("abc/123/numa_{}", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 4), (1, 0), (3, 2)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("abc/123/nu_{}_ma", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 4), (1, 0), (3, 2)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("abc/123/nu_{}_ma/546", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 4), (1, 0), (3, 2)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("abc/123/中{}文", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 4), (1, 0), (3, 2)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("/abc/123/nu_{}_ma/546/", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 4), (1, 2), (2, 3), (3, 2)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("///abc/123/nu_{}_ma/546///", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(0, 0), (1, 0), (2, 0), (3, 0)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("abc/123/nu_{}_ma/546", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: vec![(3, 0)].into_iter().collect(),
            },
            TestCase {
                shm_numa_dir_pattern: format!("abc/123/nu_{}_ma/546", NUMA_NODE_IDX_VAR_NAME),
                numa_node_files: Default::default(),
            },
        ];

        for c in cases {
            let tempdir = tempfile::tempdir().expect("Failed to create tempdir");

            let numa_node_num = *c.numa_node_files.keys().max().unwrap_or(&0) + 1;
            let mut expected_numa_shm_files = vec![vec![]; numa_node_num];
            for (numa_index, count) in c.numa_node_files {
                let dir = c.shm_numa_dir_pattern.replacen(
                    NUMA_NODE_IDX_VAR_NAME,
                    numa_index.to_string().as_str(),
                    1,
                );
                expected_numa_shm_files[numa_index] =
                    generated_random_files(tempdir.path().join(dir.trim_matches('/')), count);
            }
            if expected_numa_shm_files.iter().all(Vec::is_empty) {
                expected_numa_shm_files = Vec::new();
            }

            let p =
                ShmNumaDirPattern::new(&c.shm_numa_dir_pattern, tempdir.path().to_str().unwrap());
            let mut actually_numa_shm_files =
                scan_shm_files(&p).expect("scan shm files must be ok");
            actually_numa_shm_files
                .iter_mut()
                .for_each(|files| files.sort());

            assert_eq!(expected_numa_shm_files, actually_numa_shm_files);
        }
    }

    fn generated_random_files(dir: impl AsRef<Path>, count: usize) -> Vec<PathBuf> {
        fn filename_fn(rng: &mut ThreadRng) -> String {
            let len = rng.gen_range(1..=30);
            repeat(())
                .map(|()| rng.sample(Alphanumeric))
                .map(char::from)
                .take(len)
                .collect()
        }

        let mut rng = rand::thread_rng();
        let dir = dir.as_ref();
        fs::create_dir_all(dir).expect("Failed to create dir");

        let mut files: Vec<PathBuf> = (0..count)
            .map(|_| {
                let filename = filename_fn(&mut rng);
                let p = dir.join(filename);
                let mut data = vec![0; rng.gen_range(0..100)];
                rng.fill(data.as_mut_slice());
                fs::write(&p, &data).expect("Failed to write random data");
                p
            })
            .collect();

        files.sort();
        files
    }
}
