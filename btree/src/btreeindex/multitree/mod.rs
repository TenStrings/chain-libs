pub mod keyvalue;
pub mod multimap;

use super::page_manager::PageIdGenerator;
use super::version_management::TreeIdentifier;
use super::{Node, PageId, Pages, PagesInitializationParams, StaticSettings, NODES_PER_PAGE};
use crate::btreeindex::BTree;
use crate::{BTreeStoreError, FixedSize};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use thiserror::Error;

const STATIC_SETTINGS_FILE: &str = "static_settings";
const TREE_FILE: &str = "tree_file";

const ROOTS_FILE_PATH: &str = "roots_meta";

const NEXT_PAGE_FILE: &str = "next_page";
const FREE_PAGES_DIR_PATH: &str = "free_pages_meta";

#[derive(Error, Debug)]
pub enum TaggedTreeError {
    #[error("source tag not found")]
    SrcTagNotFound,
    #[error("destination tag is already used")]
    DstTagAlreadyExists,
}

struct PageGenerator {
    free_pages: BTree<PageId, ()>,
    next_page: AtomicU32,
    next_page_file: File,
}

#[derive(Clone)]
struct SharedPageGenerator(pub Arc<PageGenerator>);

impl PageIdGenerator for SharedPageGenerator {
    fn next_id(&self) -> PageId {
        PageGenerator::next_id(&self.0)
    }

    fn new_id(&mut self) -> PageId {
        PageGenerator::new_id(&self.0)
    }
}

impl PageGenerator {
    fn new(dir_path: impl AsRef<Path>) -> Result<PageGenerator, BTreeStoreError> {
        let free_pages = BTree::new(dir_path.as_ref().join(FREE_PAGES_DIR_PATH), 4096)?;
        let next_page = AtomicU32::new(1);

        let mut next_page_file = OpenOptions::new()
            .read(true)
            .create(true)
            .write(true)
            .open(dir_path.as_ref().join(NEXT_PAGE_FILE))?;

        next_page_file.write_u32::<LittleEndian>(1)?;

        Ok(PageGenerator {
            free_pages,
            next_page,
            next_page_file,
        })
    }

    fn open(dir_path: impl AsRef<Path>) -> Result<PageGenerator, BTreeStoreError> {
        let free_pages = BTree::open(dir_path.as_ref().join(FREE_PAGES_DIR_PATH))?;

        let mut next_page_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir_path.as_ref().join(NEXT_PAGE_FILE))?;

        let next_page = next_page_file
            .read_u32::<LittleEndian>()
            .expect("Couldn't read next page id");

        Ok(PageGenerator {
            free_pages,
            next_page: AtomicU32::new(next_page),
            next_page_file,
        })
    }

    fn next_id(&self) -> PageId {
        self.next_page.load(std::sync::atomic::Ordering::Acquire)
    }

    fn new_id(&self) -> PageId {
        let next = self.free_pages.pop_max().expect("pop max shouldn't error");

        next.map(|(key, _)| key)
            .unwrap_or_else(|| self.next_page.fetch_add(1, Ordering::Relaxed))
    }

    fn return_page(&self, page_id: PageId) -> Result<(), BTreeStoreError> {
        self.free_pages.insert_one(page_id, ())?;

        Ok(())
    }

    fn save(&self) -> Result<(), BTreeStoreError> {
        let next_page = self.next_page.load(Ordering::SeqCst);

        self.next_page_file
            .try_clone()
            .unwrap()
            .write_all(&next_page.to_le_bytes())
            .expect("Can't save next_page");

        self.free_pages.checkpoint()?;

        Ok(())
    }
}

impl TreeIdentifier for PageId {
    fn root(&self) -> PageId {
        *self
    }
}

struct TreeParams<Tag, K, V> {
    roots: BTree<Tag, PageId>,
    page_manager: SharedPageGenerator,
    static_settings: StaticSettings,
    pages: Arc<Pages>,
    phantom: std::marker::PhantomData<(K, V)>,
}

impl<Tag, K, V> TreeParams<Tag, K, V>
where
    Tag: FixedSize,
    K: FixedSize,
    V: FixedSize,
{
    fn new_tree<P>(
        dir_path: P,
        page_size: u16,
        initial_tag: Tag,
    ) -> Result<TreeParams<Tag, K, V>, BTreeStoreError>
    where
        P: AsRef<Path>,
    {
        std::fs::create_dir_all(dir_path.as_ref())?;

        let mut static_settings_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(dir_path.as_ref().join(STATIC_SETTINGS_FILE))?;

        let tree_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(dir_path.as_ref().join(TREE_FILE))?;

        let pages_storage =
            crate::storage::MmapStorage::new(tree_file, page_size as u64 * NODES_PER_PAGE)?;

        let pages = Pages::new(PagesInitializationParams {
            storage: pages_storage,
            page_size: page_size.try_into().unwrap(),
        });

        let page_manager = Arc::new(PageGenerator::new(dir_path.as_ref())?);
        let roots = BTree::new(dir_path.as_ref().join(ROOTS_FILE_PATH), 4096)?;

        let first_page_id = page_manager.new_id();

        roots.insert_one(initial_tag, first_page_id)?;

        let mut root_page = pages.mut_page(first_page_id)?;

        root_page.as_slice(|page| {
            Node::<K, &mut [u8]>::new_leaf::<V>(page);
        });

        let static_settings = StaticSettings {
            page_size,
            key_buffer_size: K::max_size().try_into().unwrap(),
        };

        static_settings.write(&mut static_settings_file)?;

        Ok(TreeParams {
            roots,
            page_manager: SharedPageGenerator(page_manager),
            static_settings,
            pages: Arc::new(pages),
            phantom: std::marker::PhantomData,
        })
    }

    pub fn open<P: AsRef<Path>>(dir_path: P) -> Result<TreeParams<Tag, K, V>, BTreeStoreError> {
        let mut static_settings_file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(dir_path.as_ref().join(STATIC_SETTINGS_FILE))?;

        let tree_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir_path.as_ref().join(TREE_FILE))?;

        let static_settings = StaticSettings::read(&mut static_settings_file)?;

        let pages_storage = crate::storage::MmapStorage::new(
            tree_file,
            static_settings.page_size as u64 * NODES_PER_PAGE,
        )?;

        let pages = Pages::new(PagesInitializationParams {
            storage: pages_storage,
            page_size: static_settings.page_size.try_into().unwrap(),
        });

        let roots = BTree::open(dir_path.as_ref().join(ROOTS_FILE_PATH))?;

        let page_manager = Arc::new(PageGenerator::open(dir_path.as_ref())?);

        Ok(TreeParams {
            roots,
            page_manager: SharedPageGenerator(page_manager),
            static_settings,
            pages: Arc::new(pages),
            phantom: std::marker::PhantomData,
        })
    }
}
