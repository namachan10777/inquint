//! A concurrent append-only slab: lock-free reads by index, with slots
//! published through owner-exclusive writes.
//!
//! Storage is a fixed table of geometrically growing segments, so a slot's
//! address never moves — readers index without any lock. The intended
//! protocol (used by the interners):
//!
//! 1. the writer claims a unique index (e.g. from an atomic counter held
//!    under the interner's shard lock),
//! 2. writes the slot with [`Slab::write`],
//! 3. publishes the index with a release operation (the shard lock's
//!    unlock, or any release store the reader later acquires).
//!
//! [`Slab::get`] on an index that was never published this way is
//! undefined behavior territory — hence both methods are honest about
//! their contracts. All unsafety of the concurrent stores is contained in
//! this module.

use std::alloc::{alloc_zeroed, Layout};
use std::sync::atomic::{AtomicPtr, Ordering};

/// log2 of segment 0's slot count.
const SEG0_SHIFT: u32 = 12; // 4096
/// Number of segments: capacity = 4096 * (2^28 - 1) ≈ 2^40 slots — far
/// above the 2^31 id space, so `locate` never runs off the table.
const NSEG: usize = 28;

pub struct Slab<T> {
    segments: [AtomicPtr<T>; NSEG],
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// (segment, offset) of slot `i`: segment `k` holds `4096 << k` slots.
#[inline]
fn locate(i: u32) -> (usize, usize) {
    let j = (i >> SEG0_SHIFT) + 1;
    let seg = (31 - j.leading_zeros()) as usize;
    let base = ((1usize << seg) - 1) << SEG0_SHIFT;
    (seg, i as usize - base)
}

impl<T> Slab<T> {
    pub const fn new() -> Self {
        Slab {
            segments: [const { AtomicPtr::new(std::ptr::null_mut()) }; NSEG],
        }
    }

    /// Read slot `i`.
    ///
    /// The slot must have been written by [`Slab::write`] and published to
    /// this thread by a release/acquire edge (see the module docs); the
    /// returned reference is valid forever (segments are never freed).
    #[inline]
    pub fn get(&self, i: u32) -> &T {
        let (seg, off) = locate(i);
        // Relaxed is sound here: the publication contract (module docs)
        // means the reader acquired an edge ordered after both the
        // segment-pointer CAS and the slot write — the pointer it reads
        // is therefore non-null and the slot contents visible.
        let ptr = self.segments[seg].load(Ordering::Relaxed);
        debug_assert!(!ptr.is_null(), "slab read of unpublished segment");
        // Safety: segment pointers are only ever null → valid (CAS'd once,
        // never freed), and `off` is in bounds by construction.
        unsafe { &*ptr.add(off) }
    }

    /// Write slot `i`. The caller must be the unique owner of index `i`
    /// (it claimed it from the id allocator and nobody else writes or
    /// reads it until publication).
    ///
    /// # Safety
    /// - exclusive ownership of slot `i` as described above;
    /// - `T`'s drop is never run (the slab leaks — fine for interners).
    pub unsafe fn write(&self, i: u32, value: T) {
        let (seg, off) = locate(i);
        let ptr = self.ensure_segment(seg);
        std::ptr::write(ptr.add(off), value);
    }

    /// Get or allocate segment `seg` (zero-initialized; losers of the
    /// allocation race free theirs).
    fn ensure_segment(&self, seg: usize) -> *mut T {
        let cur = self.segments[seg].load(Ordering::Acquire);
        if !cur.is_null() {
            return cur;
        }
        let len = (1usize << SEG0_SHIFT) << seg;
        let layout = Layout::array::<T>(len).expect("slab segment layout");
        // Zeroed so torn reads of never-written slots at least don't read
        // uninitialized memory in debug tooling; the protocol forbids such
        // reads anyway.
        let fresh = unsafe { alloc_zeroed(layout) } as *mut T;
        assert!(!fresh.is_null(), "slab segment allocation failed");
        match self.segments[seg].compare_exchange(
            std::ptr::null_mut(),
            fresh,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => fresh,
            Err(winner) => {
                unsafe { std::alloc::dealloc(fresh as *mut u8, layout) };
                winner
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_covers_boundaries() {
        assert_eq!(locate(0), (0, 0));
        assert_eq!(locate(4095), (0, 4095));
        assert_eq!(locate(4096), (1, 0));
        assert_eq!(locate(4096 + 8191), (1, 8191));
        assert_eq!(locate(4096 * 3), (2, 0));
    }

    #[test]
    fn write_then_get() {
        let slab: Slab<u64> = Slab::new();
        for i in 0..20000u32 {
            unsafe { slab.write(i, i as u64 * 3) };
        }
        for i in 0..20000u32 {
            assert_eq!(*slab.get(i), i as u64 * 3);
        }
    }
}
