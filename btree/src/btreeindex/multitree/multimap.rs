use super::{SharedPageGenerator, TaggedTreeError};
use crate::btreeindex::node::NodeRef;
use crate::btreeindex::page_manager::PageIdGenerator;
use crate::btreeindex::transaction::{CommitResult, ReadTransaction, WriteTransaction};
use crate::btreeindex::{
    backtrack::UpdateBacktrack, tree_algorithm, Node, PageId, Pages, StaticSettings,
};
use crate::btreeindex::{BTree, BTreeIterator};
use crate::{BTreeStoreError, FixedSize};
use std::borrow::Borrow;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::path::Path;
use std::sync::Arc;

pub struct MultiMap<Tag, K, V>
where
    Tag: FixedSize,
    K: FixedSize,
    V: FixedSize,
{
    roots: BTree<Tag, PageId>,
    page_manager: SharedPageGenerator,
    static_settings: StaticSettings,
    pages: Arc<Pages>,
    phantom: PhantomData<(Tag, K, V)>,
}

impl<Tag, K, V> MultiMap<Tag, K, V>
where
    Tag: FixedSize,
    K: FixedSize,
    V: FixedSize,
{
    pub fn new(
        dir_path: impl AsRef<Path>,
        page_size: u16,
        initial_tag: Tag,
    ) -> Result<MultiMap<Tag, K, V>, BTreeStoreError> {
        let super::TreeParams {
            roots,
            page_manager,
            static_settings,
            pages,
            ..
        } = super::TreeParams::<Tag, K, V>::new_tree(dir_path, page_size, initial_tag)?;

        Ok(MultiMap {
            roots,
            page_manager,
            static_settings,
            pages,
            phantom: PhantomData,
        })
    }

    pub fn open(dir_path: impl AsRef<Path>) -> Result<MultiMap<Tag, K, V>, BTreeStoreError> {
        let super::TreeParams {
            roots,
            page_manager,
            static_settings,
            pages,
            ..
        } = super::TreeParams::<Tag, K, V>::open(dir_path)?;

        Ok(MultiMap {
            roots,
            page_manager,
            static_settings,
            pages,
            phantom: PhantomData,
        })
    }

    pub fn write(&self, from: Tag, to: Tag) -> Result<WriteTx<Tag, K, V>, BTreeStoreError> {
        let root = self
            .roots
            .get(&from, |page_id| page_id.cloned())
            .clone()
            .ok_or(TaggedTreeError::SrcTagNotFound)?;

        let page_size = usize::try_from(self.static_settings.page_size).unwrap();

        Ok(WriteTx {
            tx: WriteTransaction::new(root, &self.pages, self.page_manager.clone()),
            roots: &self.roots,
            page_size,
            to,
            pages: Arc::clone(&self.pages),
            page_manager: self.page_manager.clone(),
            phantom: PhantomData,
        })
    }

    pub fn read(&self, tag: Tag) -> Result<Option<ReadTx<K>>, BTreeStoreError> {
        self.roots
            .get(&tag, |root| {
                root.map(|root| {
                    Ok(ReadTx {
                        tx: ReadTransaction::new(*root, Arc::clone(&self.pages)),
                        pages: Arc::clone(&self.pages),
                        phantom_keys: PhantomData,
                    })
                })
            })
            .transpose()
    }

    pub fn sync(&self) -> Result<(), BTreeStoreError> {
        self.pages.sync_file()?;
        self.roots.checkpoint()?;
        self.page_manager.0.save()
    }
}

pub struct WriteTx<'a, 'b, Tag: FixedSize, K: FixedSize, V: FixedSize> {
    tx: WriteTransaction<'a, SharedPageGenerator>,
    page_size: usize,
    roots: &'b BTree<Tag, PageId>,
    to: Tag,
    pages: Arc<Pages>,
    page_manager: SharedPageGenerator,
    phantom: PhantomData<(K, V)>,
}

pub struct ReadTx<K: FixedSize> {
    tx: ReadTransaction<PageId, Arc<Pages>>,
    pages: Arc<Pages>,
    phantom_keys: PhantomData<K>,
}

impl<'a, 'b, Tag: FixedSize, K: FixedSize, V: FixedSize> WriteTx<'a, 'b, Tag, K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Result<(), BTreeStoreError> {
        let subroot = {
            let read_tx = ReadTransaction::new(self.tx.root(), Arc::clone(&self.pages));
            let page_ref = tree_algorithm::search::<PageId, K, _, _>(&read_tx, &key);

            page_ref.as_node(|node: Node<K, &[u8]>| -> Result<PageId, BTreeStoreError> {
                match node.as_leaf::<PageId>().keys().binary_search::<K>(&key) {
                    Ok(pos) => Ok(*node.as_leaf::<PageId>().values().get(pos).borrow()),
                    Err(_) => {
                        let new_root_page = self.page_manager.new_id();

                        let mut root_page = self.pages.mut_page(new_root_page)?;

                        root_page.as_slice(|page| {
                            Node::<K, &mut [u8]>::new_leaf::<V>(page);
                        });

                        Ok(new_root_page)
                    }
                }
            })?
        };

        let new_root = {
            let mut subtx = self.tx.sub(subroot);

            tree_algorithm::append(
                &mut subtx,
                |max_key: Option<u64>| max_key.map(|k| k + 1).unwrap_or(0),
                value,
                self.page_size,
            )?;

            match subtx.commit::<u64>() {
                CommitResult::RootTx(_) => unreachable!(),
                CommitResult::SubTx(new_root) => new_root,
            }
        };

        let updated = UpdateBacktrack::new_search_for(&mut self.tx, &key).update(|_| new_root);

        if let Err(BTreeStoreError::KeyNotFound) = updated {
            tree_algorithm::insert(&mut self.tx, key, new_root, self.page_size)?;
        }

        Ok(())
    }

    pub fn commit(self) -> Result<ReadTx<K>, BTreeStoreError> {
        let delta = match self.tx.commit::<u64>() {
            CommitResult::RootTx(delta) => delta,
            CommitResult::SubTx(_) => unreachable!(),
        };

        self.roots
            .insert_one(self.to, delta.new_root)
            .map_err(|err| match err {
                BTreeStoreError::DuplicatedKey => {
                    BTreeStoreError::TaggedTree(TaggedTreeError::DstTagAlreadyExists)
                }
                err => err,
            })?;

        Ok(ReadTx {
            tx: ReadTransaction::new(delta.new_root, Arc::clone(&self.pages)),
            pages: Arc::clone(&self.pages),
            phantom_keys: PhantomData,
        })
    }
}

pub type MultiMapKeyIterator<'a, R, V> = BTreeIterator<PageId, R, u64, u64, V, &'a Pages>;

impl<K: FixedSize> ReadTx<K> {
    pub fn get<V, Q, F, R, Ret>(&self, key: &Q, range: R, f: F) -> Ret
    where
        Q: Ord,
        K: Borrow<Q>,
        R: RangeBounds<u64>,
        V: FixedSize,
        F: FnOnce(Option<MultiMapKeyIterator<R, V>>) -> Ret,
    {
        let page_ref = tree_algorithm::search::<PageId, K, Q, Arc<Pages>>(&self.tx, key);

        page_ref.as_node(|node: Node<K, &[u8]>| {
            match node.as_leaf::<PageId>().keys().binary_search::<Q>(key) {
                Ok(pos) => {
                    let root = *node.as_leaf::<PageId>().values().get(pos).borrow();
                    let tx = ReadTransaction::new(root, self.pages.borrow());
                    let iter = BTreeIterator::new(tx, range);
                    f(Some(iter))
                }
                Err(_) => f(None),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::U64Key;
    use tempfile::tempdir;

    extern crate rand;
    extern crate tempfile;

    const INITIAL_TAG: u64 = 0;

    fn new_tree() -> MultiMap<U64Key, U64Key, U64Key> {
        let dir_path = tempdir().unwrap();

        let page_size = 88;

        let tree: MultiMap<U64Key, U64Key, U64Key> =
            MultiMap::new(dir_path.path(), page_size, U64Key(INITIAL_TAG)).unwrap();

        tree
    }

    #[test]
    fn branches_are_different() {
        let tree = new_tree();

        let tag1 = U64Key(1);
        let tag2 = U64Key(2);
        {
            let mut wtx = tree.write(U64Key(INITIAL_TAG), tag1.clone()).unwrap();

            wtx.insert(U64Key(3), U64Key(4)).unwrap();
            wtx.insert(U64Key(3), U64Key(5)).unwrap();
            wtx.commit().unwrap();
        }
        {
            let mut wtx = tree.write(U64Key(INITIAL_TAG), tag2.clone()).unwrap();

            wtx.insert(U64Key(3), U64Key(5)).unwrap();
            wtx.insert(U64Key(3), U64Key(6)).unwrap();
            wtx.commit().unwrap();
        }

        let rtx1 = tree.read(tag1).unwrap().unwrap();
        let rtx2 = tree.read(tag2).unwrap().unwrap();
        let v1: Vec<U64Key> = rtx1.get(&U64Key(3), .., |iter| iter.unwrap().collect());
        let v2: Vec<U64Key> = rtx2.get(&U64Key(3), .., |iter| iter.unwrap().collect());

        assert_ne!(v1, v2);

        assert_eq!(v1, vec![U64Key(4), U64Key(5)]);
        assert_eq!(v2, vec![U64Key(5), U64Key(6)]);
    }

    use model::*;
    #[quickcheck]
    fn test_append(ops: Vec<Op>) -> bool {
        let db = new_tree();
        let mut reference = Reference::default();

        for op in ops {
            match op {
                Op::Write { from, to, op } => {
                    let ref_result = reference.write(from, to, |old| {
                        let mut new = old.clone();
                        match op {
                            WriteOp::Append { key, ref elements } => {
                                if let Some(value) = new.get_mut(&key) {
                                    for e in elements {
                                        value.push(*e);
                                    }
                                } else {
                                    new.insert(key, elements.clone());
                                }
                            }
                        }

                        new
                    });

                    let wtx = db.write(from.into(), to.into());

                    let mut wtx = if let Err(TaggedTreeError::SrcTagNotFound) = ref_result {
                        assert!(wtx.is_err());
                        continue;
                    } else {
                        wtx.unwrap()
                    };

                    match op {
                        WriteOp::Append { key, elements } => {
                            for e in elements {
                                wtx.insert(U64Key(key), U64Key(e)).unwrap();
                            }
                        }
                    }
                    let commit = wtx.commit();

                    if let Err(TaggedTreeError::DstTagAlreadyExists) = ref_result {
                        assert!(commit.is_err());
                        continue;
                    } else {
                        assert!(commit.is_ok());
                    };
                }
                Op::Read { from, keys } => {
                    let rtx = db.read(from.into()).unwrap();
                    let pair = reference.read(from).map(|from| (from, rtx.unwrap()));

                    if let Some((refversion, dbversion)) = pair {
                        for k in keys {
                            let r = refversion.get(&k).cloned().unwrap_or(vec![]);
                            let d = dbversion.get(&U64Key(k), .., |iter| -> Vec<u64> {
                                let mut c = vec![];

                                if let Some(iter) = iter {
                                    for v in iter {
                                        c.push(v)
                                    }
                                }

                                c
                            });

                            assert_eq!(r, d);
                        }
                    }
                }
            }
        }

        true
    }

    mod model {
        use super::super::TaggedTreeError;
        use quickcheck::{Arbitrary, Gen};
        use rand::Rng;
        use std::collections::BTreeMap;

        pub struct Reference {
            versions: BTreeMap<u64, BTreeMap<u64, Vec<u64>>>,
        }

        impl Reference {
            pub fn write<F>(
                &mut self,
                from: u64,
                to: u64,
                f: F,
            ) -> Result<&BTreeMap<u64, Vec<u64>>, TaggedTreeError>
            where
                F: Fn(&BTreeMap<u64, Vec<u64>>) -> BTreeMap<u64, Vec<u64>>,
            {
                let base = &self
                    .versions
                    .get(&from)
                    .ok_or(TaggedTreeError::SrcTagNotFound)?;

                if self.versions.get(&to).is_some() {
                    return Err(TaggedTreeError::DstTagAlreadyExists);
                }

                let new = f(&base);
                self.versions.insert(to, new);

                Ok(&self.versions[&to])
            }

            pub fn read(&mut self, tag: u64) -> Option<&BTreeMap<u64, Vec<u64>>> {
                self.versions.get(&tag)
            }
        }

        impl Default for Reference {
            fn default() -> Reference {
                let mut versions = <BTreeMap<u64, BTreeMap<u64, Vec<u64>>>>::new();

                versions.insert(super::INITIAL_TAG, <BTreeMap<u64, Vec<u64>>>::new());

                Reference { versions }
            }
        }

        const MAX_TAG: u64 = 25;
        const MAX_ELEMENT: u64 = 25;
        const MAX_ELEMENTS: usize = 5;
        const MAX_KEY: u64 = 25;

        #[derive(Clone, Debug)]
        pub enum Op {
            Write { from: u64, to: u64, op: WriteOp },
            Read { from: u64, keys: Vec<u64> },
        }

        #[derive(Clone, Debug)]
        pub enum WriteOp {
            Append { key: u64, elements: Vec<u64> },
        }

        impl Arbitrary for Op {
            fn arbitrary<G: Gen>(g: &mut G) -> Op {
                match g.gen_range(0, 2) {
                    0 => {
                        let from = g.gen_range(0, MAX_TAG);
                        let to = g.gen_range(0, MAX_TAG);

                        let op = <WriteOp as Arbitrary>::arbitrary(g);

                        Op::Write { from, to, op }
                    }

                    1 => {
                        let from = g.gen_range(0, MAX_TAG);
                        let keys = Vec::<u64>::arbitrary(g);

                        Op::Read { from, keys }
                    }
                    _ => unreachable!(),
                }
            }
        }

        impl Arbitrary for WriteOp {
            fn arbitrary<G: Gen>(g: &mut G) -> WriteOp {
                let key = g.gen_range(0, MAX_KEY);
                let elements_len = g.gen_range(0, MAX_ELEMENTS);
                let mut elements = vec![];

                for _ in 0..elements_len {
                    elements.push(g.gen_range(0, MAX_ELEMENT));
                }

                WriteOp::Append { key, elements }
            }
        }
    }
}
