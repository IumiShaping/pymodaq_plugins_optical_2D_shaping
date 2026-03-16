from typing import Union, List, Tuple, TYPE_CHECKING, Iterable

import xml.etree.ElementTree as ET

from pymodaq_data.h5modules.backends import GROUP, Node
from pymodaq_data.h5modules.data_saving import DataToExportEnlargeableSaver, DataToExport, DataToExportSaver

from pymodaq.utils.h5modules.module_saving import ModuleSaver, GroupModuleType

from aenum import extend_enum

from pymodaq_gui.h5modules.saving import H5Saver

extend_enum(GroupModuleType, 'SHAPING')


class ShapingSaver(ModuleSaver):
    group_type = GroupModuleType.SHAPING

    def __init__(self):
        self._module = None
        self._module_group: GROUP = None
        self._h5saver: H5Saver = None

        self.enl_axis_names = ('step',)
        self.enl_axis_units = ('',)

        self._datatoexport_saver: DataToExportEnlargeableSaver = None

    def update_after_h5changed(self):
        self._datatoexport_saver = DataToExportEnlargeableSaver(
            self.h5saver, self.enl_axis_names, self.enl_axis_units)

    def get_set_node(self, where: Union[Node, str] = None, new=False,
                     settings_as_xml= None, common_data: DataToExport = None) -> GROUP:
        """Get the last group scan node

        Get the last Scan Group or create one
        get the last Scan Group if:
        * there is one already created
        * new is False

        Parameters
        ----------
        where: Union[Node, str]
            the path of a given node or the node itself
        new: bool
        settings_as_xml: optional, binary string to be added for new node
        common_data: optional, DataToExport
        Returns
        -------
        GROUP: the GROUP associated with this module
        """

        self._module_group = self.get_last_node(where)
        new = new or (self._module_group is None)
        if new:
            self._module_group = self._add_module(where, settings_as_xml=settings_as_xml)
        if common_data is not None:
            with DataToExportSaver(self._h5saver) as saver:
                saver.add_data(self._module_group, common_data)

        return self._module_group

    def _add_module(self, where=None, metadata=None, settings_as_xml='') -> Node:
        if metadata is None:
            metadata = {}
        if where is None:
            where = self._h5saver.raw_group


        return self._h5saver.add_incremental_group(self.group_type, where, title='BeamShaping',
                                                   settings_as_xml=settings_as_xml, metadata=metadata)


    def add_data(self, data: DataToExport):

        self._datatoexport_saver.add_data(self.get_last_node(self._h5saver.raw_group),
                                          data, axis_values=(self.h5saver.settings['N_saved'],))
