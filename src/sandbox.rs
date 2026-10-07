//! Copy-on-write-ish isolation for parallel subagents. Each gets a private
//! copy of the working tree to edit; after it finishes, only the files it
//! changed are copied back. Two subagents that touched the same file conflict
//! and neither write is applied — that's the whole point: concurrent edits in
//! one shared tree can silently clobber each other, and this makes them safe.
//!
//! No git required: the snapshot/diff is plain content hashing, and the copy
//! respects the same skip rules as search (`.`-dirs, target/, node_modules, …),
//! so a subagent sees exactly what grep/glob would.

use crate::tools::{hash, walk};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A relative path, `/`-separated and without a leading `./`, used as a stable
/// key across the main tree and a sandbox (which have different roots).
fn rel(path: &Path, root: &Path) -> String {
    let p = path.strip_prefix(root).unwrap_or(path);
    let s = p.to_string_lossy().replace('\\', "/");
    s.strip_prefix("./").unwrap_or(&s).to_string()
}

/// Content hash of every (non-skipped) file under `root`, keyed by relative
/// path. A sandbox is diffed against this to find what the subagent changed.
pub fn snapshot(root: &Path) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for e in walk(&root.to_string_lossy()) {
        if let Ok(text) = fs::read_to_string(e.path()) {
            map.insert(rel(e.path(), root), hash(&text));
        }
    }
    map
}

/// What a subagent changed in its sandbox, relative to the tree it was seeded
/// from. `changed` = added or modified (needs copy-back); `deleted` = removed.
#[derive(Default)]
pub struct Changes {
    pub changed: Vec<String>,
    pub deleted: Vec<String>,
}

impl Changes {
    /// Every path this touched — for conflict detection across sandboxes.
    fn touched(&self) -> impl Iterator<Item = &String> {
        self.changed.iter().chain(self.deleted.iter())
    }
}

/// A private copy of the working tree. The directory is removed on drop, so a
/// sandbox must outlive the merge-back that reads from it.
pub struct Sandbox {
    pub dir: PathBuf,
}

impl Sandbox {
    /// Copy every non-skipped file under `src` into a fresh temp directory.
    pub fn create(src: &Path) -> io::Result<Sandbox> {
        static N: AtomicU64 = AtomicU64::new(0);
        let uniq = format!("tern-wt-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
        let dir = std::env::temp_dir().join(uniq);
        fs::create_dir_all(&dir)?;
        for e in walk(&src.to_string_lossy()) {
            let dest = dir.join(rel(e.path(), src));
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(e.path(), &dest)?;
        }
        Ok(Sandbox { dir })
    }

    /// Diff this sandbox against the snapshot it was seeded from.
    pub fn changes(&self, base: &HashMap<String, u64>) -> Changes {
        let mut c = Changes::default();
        let mut present = HashSet::new();
        for e in walk(&self.dir.to_string_lossy()) {
            let key = rel(e.path(), &self.dir);
            present.insert(key.clone());
            let Ok(text) = fs::read_to_string(e.path()) else { continue };
            if base.get(&key) != Some(&hash(&text)) {
                c.changed.push(key);
            }
        }
        for key in base.keys() {
            if !present.contains(key) {
                c.deleted.push(key.clone());
            }
        }
        c
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

/// Paths changed by more than one sandbox in the same batch: applying either
/// would clobber the other, so neither is applied.
pub fn conflicts(per: &[(usize, Changes)]) -> HashSet<String> {
    let mut seen = HashSet::new();
    let mut dup = HashSet::new();
    for (_, c) in per {
        for p in c.touched() {
            if !seen.insert(p.clone()) {
                dup.insert(p.clone());
            }
        }
    }
    dup
}

/// Copy a sandbox's non-conflicting changes back into `dest` (the main tree),
/// applying deletions too. Returns the conflicting paths it refused to apply.
pub fn merge(sandbox: &Path, c: &Changes, conflicts: &HashSet<String>, dest: &Path) -> io::Result<Vec<String>> {
    let mut skipped = Vec::new();
    for p in &c.changed {
        if conflicts.contains(p) {
            skipped.push(p.clone());
            continue;
        }
        let to = dest.join(p);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(sandbox.join(p), to)?;
    }
    for p in &c.deleted {
        if conflicts.contains(p) {
            skipped.push(p.clone());
        } else {
            let _ = fs::remove_file(dest.join(p));
        }
    }
    Ok(skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("tern-test-{}-{}", std::process::id(), rand()));
        fs::create_dir_all(&d).unwrap();
        d
    }
    fn rand() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
    }

    #[test]
    fn sandbox_copies_tree_and_diffs_changes() {
        let src = tmp();
        fs::write(src.join("keep.txt"), "same").unwrap();
        fs::write(src.join("edit.txt"), "before").unwrap();
        fs::write(src.join("gone.txt"), "x").unwrap();

        let base = snapshot(&src);
        let sb = Sandbox::create(&src).unwrap();
        // Edit one file, add one, delete one — inside the sandbox.
        fs::write(sb.dir.join("edit.txt"), "after").unwrap();
        fs::write(sb.dir.join("new.txt"), "hi").unwrap();
        fs::remove_file(sb.dir.join("gone.txt")).unwrap();

        let c = sb.changes(&base);
        let mut changed = c.changed.clone();
        changed.sort();
        assert_eq!(changed, vec!["edit.txt".to_string(), "new.txt".to_string()]);
        assert_eq!(c.deleted, vec!["gone.txt".to_string()]);

        // Merge back with no conflicts applies everything.
        let skipped = merge(&sb.dir, &c, &HashSet::new(), &src).unwrap();
        assert!(skipped.is_empty());
        assert_eq!(fs::read_to_string(src.join("edit.txt")).unwrap(), "after");
        assert_eq!(fs::read_to_string(src.join("new.txt")).unwrap(), "hi");
        assert!(!src.join("gone.txt").exists());
        fs::remove_dir_all(&src).ok();
    }

    #[test]
    fn overlapping_edits_conflict_and_are_not_applied() {
        let a = Changes { changed: vec!["shared.rs".into(), "a_only.rs".into()], deleted: vec![] };
        let b = Changes { changed: vec!["shared.rs".into(), "b_only.rs".into()], deleted: vec![] };
        let per = vec![(0usize, a), (1usize, b)];
        let con = conflicts(&per);
        assert!(con.contains("shared.rs") && con.len() == 1);

        // The conflicting file is reported skipped; the private one still applies.
        let src = tmp();
        let sb = Sandbox::create(&src).unwrap();
        fs::write(sb.dir.join("shared.rs"), "x").unwrap();
        fs::write(sb.dir.join("a_only.rs"), "y").unwrap();
        let skipped = merge(&sb.dir, &per[0].1, &con, &src).unwrap();
        assert_eq!(skipped, vec!["shared.rs".to_string()]);
        assert_eq!(fs::read_to_string(src.join("a_only.rs")).unwrap(), "y");
        assert!(!src.join("shared.rs").exists()); // conflicting write withheld
        fs::remove_dir_all(&src).ok();
    }
}
