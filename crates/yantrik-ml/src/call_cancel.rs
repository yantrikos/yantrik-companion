//! A cancel flag for the model request running on this thread, set by whoever knows the work it
//! serves is over. Yantrik Mind E.VOICE2: a person who talks over a spoken answer has ended the turn,
//! and the streamed request behind it held the only model slot until the model finished writing an
//! answer nobody would hear.
//!
//! Shaped like [`crate::call_timeout::install_call_cap`]: installed per job on a (pooled) blocking
//! thread, restored when the guard drops. No flag installed -- every caller that never asks for
//! one -- means no behaviour change.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

thread_local! {
    static CANCEL: std::cell::RefCell<Option<Arc<AtomicBool>>> = const { std::cell::RefCell::new(None) };
}

/// Restores the flag that was in force before [`install_cancel`], when dropped.
pub struct CancelGuard(Option<Arc<AtomicBool>>);

impl Drop for CancelGuard {
    fn drop(&mut self) {
        let prev = self.0.take();
        CANCEL.with(|c| *c.borrow_mut() = prev);
    }
}

/// Watch `flag` for every model request made on this thread until the guard drops. `None` installs
/// "no flag" for the guard's lifetime, so a flag left by an earlier job cannot leak into this one.
pub fn install_cancel(flag: Option<Arc<AtomicBool>>) -> CancelGuard {
    CancelGuard(CANCEL.with(|c| c.replace(flag)))
}

/// True when this thread's flag is set.
pub fn cancelled() -> bool {
    CANCEL.with(|c| c.borrow().as_ref().is_some_and(|f| f.load(Ordering::SeqCst)))
}

/// The error a cancelled request returns. TYPED, so a wrapper can tell a cancel from a failure:
/// a cancel is not the backend's fault, and must not count against it or start a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled: the work this request served is over")
    }
}

impl std::error::Error for Cancelled {}

/// [`Cancelled`] when this thread's flag is set; checked between the lines of a stream.
pub fn check() -> anyhow::Result<()> {
    if cancelled() {
        return Err(anyhow::Error::new(Cancelled));
    }
    Ok(())
}

/// Whether `e` is (or wraps, through any `.context`) a cancel.
pub fn is_cancel(e: &anyhow::Error) -> bool {
    e.downcast_ref::<Cancelled>().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flag_is_seen_only_while_installed_and_the_thread_is_left_as_found() {
        assert!(check().is_ok(), "no flag: never cancelled");
        let flag = Arc::new(AtomicBool::new(false));
        {
            let _g = install_cancel(Some(flag.clone()));
            assert!(!cancelled());
            flag.store(true, Ordering::SeqCst);
            assert!(cancelled());
            let e = check().unwrap_err();
            assert!(format!("{e:#}").contains("cancelled"));
            assert!(is_cancel(&e), "typed");
            assert!(is_cancel(&e.context("reading SSE line")), "typed, through context");
            assert!(!is_cancel(&anyhow::anyhow!("cancelled by the server")), "the word alone is not a cancel");
            {
                let _inner = install_cancel(None);
                assert!(!cancelled(), "an inner job with no flag is not cancelled");
            }
            assert!(cancelled(), "the outer flag is back");
        }
        assert!(!cancelled(), "dropped: the thread is as it was");
    }
}
