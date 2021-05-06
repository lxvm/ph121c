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
            for key, value in metadata.items():
                grp.attrs[key] = value
            for i, e in enumerate(data):
                # do not rewrite metadata to subsets
                rwrite(grp, str(i), e, {})
                
    with h5py.File(archive, 'a') as f:
        try:
            rwrite(f, path, data, metadata)
        except Exception as ex:
            raise UserWarning('Couldnt write dataset. Check data items', ex)
            
def inquire (archive, path='/', fields=True, select=True):
    """Retrieve metadata from one or more datasets."""
    def rinquire (obj, fields, select):
        """Recursively inquire"""
        def get_chosen_attrs (obj, fields, select):
            """Retrieve the chosen attributes"""
            if select:
                if isinstance(select, dict):
                    for k, v in select.items():
                        if k in obj:
                            if v not in obj[k]:
                                selected = False
                                break
                else:
                    selected = True
            else:
                selected = False
            if selected:
                if fields == True:
                    return dict(obj.attrs)
                return { 
                    field : obj.attrs[field] 
                    for field in fields
                    if field in obj.attrs
                }
            
        if isinstance(obj, h5py.Dataset):
            return get_chosen_attrs(obj, fields, select)
        elif isinstance(obj, h5py.Group):
            attrs = { item : rinquire(obj[item], fields, select) for item in obj }
            attrs.update(get_chosen_attrs(obj, fields, select))
            return attrs
        
    with h5py.File(archive, 'r') as f:
        return rinquire(f[path], fields, select)
            
            