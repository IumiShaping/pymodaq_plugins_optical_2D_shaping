
from pymodaq_data.h5modules.data_saving import DataLoader

with DataLoader('saved_target.h5field') as dl:
    print(f'Structure for a *.h5field file')
    for node in dl.walk_nodes('/'):
        print(f'Node: {node}')
        for attr in node.attrs:
            print(f"    - Attr list: {attr}, {node.attrs[attr]}")


with DataLoader('shaped_gbsax.h5beam') as dl:
    print(f'Structure for a *.h5beam file')
    for node in dl.walk_nodes('/'):
        print(f'Node: {node}')
        for attr in node.attrs:
            print(f"    - Attr list: {attr}, {node.attrs[attr]}")