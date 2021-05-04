"""Stores/fetches datasets from HDF5 archive to/from numpy arrays at job level.

Storing interface: pass a path unique to the job being saved, the object,
metadata to add to that, and the archive to use

If an object is a numpy dtype that h5py can store, then a dataset is saved 
directly to the path. Otherwise, it is assumed to be an iterable of numpy
objects that can be saved and a group is created every for every iterable,
with the elements of the iterable saved as datasets (or groups if they can't 
be saved as datasets). This works recursively.

Fetching interface: Pass the path whose contents are desired and the archive

The fetching interface also works recursively by gathering all the datasets
as numpy arrays and saving them into tuples based on the groups they are stored.
Note that this means if you pass '/' the root group, all the archive will be 
extracted!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Metadata is ignored in the current implementation of `find` but could be added.
"""

import h5py

def find (path, archive):
    """Return requested file (will raise error if not there)."""
    with h5py.File(archive, 'r') as f:
        dset = f[path]
        if isinstance(dset, h5py.Group):
            return tuple(find('/'.join([path, item]), archive) for item in dset)
        elif isinstance(dset, h5py.Dataset):
            return dset[:]

def save (path, data, metadata, archive):
    """Store a dataset."""
    def rwrite (obj, path, data, metadata):
        """Recursively write to a dataset."""
        try:
            dset = obj.create_dataset(path, data=data)
            for key, value in metadata.items():
                dset.attrs[key] = value
        except TypeError:
            grp = obj.create_group(path)
            for i, e in enumerate(data):
                rwrite(grp, str(i), e, metadata)
                
    with h5py.File(archive, 'a') as f:
        try:
            rwrite(f, path, data, metadata)
        except Exception as ex:
            raise UserWarning('Couldnt write dataset. Check data items', ex)