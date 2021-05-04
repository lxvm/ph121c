"""Stores, and fetches datasets from HDF5 archive from numpy arrays.
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
    def write (obj, path, data, metadata):
        """Write to a dataset."""
        dset = obj. create_dataset(path, data=data)
        for key, value in metadata.items():
            dset.attrs[key] = value
            
    with h5py.File(archive, 'a') as f:
        try:
            write(f, path, data, metadata)
        except TypeError:
            try:
                grp = f.create_group(path)
                for i, e in enumerate(data):
                    write(grp, str(i), e, dict(**metadata, item=i))
            except Exception as ex:
                raise UserWarning('Couldnt write dataset. Check data items', ex)