import h5py

# Replace 'dataset/episode_0.hdf5' with your actual file path
file_path = 'dataset/episode_0.hdf5'

with h5py.File(file_path, 'r') as f:
    def explore_hdf5(group, indent=0):
        """Recursively explore the structure of an HDF5 group or file."""
        for key in group.keys():
            item = group[key]
            print(' ' * indent + f"{key}: {type(item)}")
            if isinstance(item, h5py.Group):
                # Recurse into groups
                explore_hdf5(item, indent + 2)
            elif isinstance(item, h5py.Dataset):
                # Display dataset info
                print(' ' * (indent + 2) + f"Shape: {item.shape}, Dtype: {item.dtype}")

    # Start exploring from the root of the file
    print("File attributes:")
    for attr_name, attr_value in f.attrs.items():
        print(f"  {attr_name}: {attr_value}")

    print("\nFile structure:")
    explore_hdf5(f)