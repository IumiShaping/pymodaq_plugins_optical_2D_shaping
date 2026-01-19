from dataclasses import dataclass

from qtpy.QtCore import QObject, Signal

from pymodaq_gui.parameter.pymodaq_ptypes import registerParameterType, GroupParameter


from pymodaq_gui.managers.parameter_manager import ParameterManager

from pymodaq_plugins_optical_2D_shaping import Config as PluginConfig
from pymodaq_plugins_optical_2D_shaping.setup.factory import ElementFactory


element_factory = ElementFactory()


class ScalableSetupElement(GroupParameter):
    """
    """

    def __init__(self, **opts):
        opts['type'] = 'group_setup'
        opts['addText'] = "Add"
        opts['addList'] = element_factory.get_elements()
        super().__init__(**opts)

    def addNew(self, typ: str):
        """
        """
        name_prefix = 'element'
        new_index = len(self.childs)+1
        element = element_factory.create_element(typ, new_index)

        child = {
            'title': f'Elt {new_index}',
            'name': f'{name_prefix}_{new_index}',
            'type': 'group',
            'removable': True,
            'children': element.to_list()
        }
        self.addChild(child)

registerParameterType('group_setup', ScalableSetupElement, override=True)



class Setup(QObject, ParameterManager):
    params = [
        {'title': 'Elements', 'name': 'elements', 'type': ScalableSetupElement},
    ]

    def __init__(self, ):
        super().__init__()

    def show(self, show=True):
        self.settings_tree.setVisible(show)



if __name__ == '__main__':
    from pymodaq_gui.qt_utils import mkQApp

    app = mkQApp('Setup')

    setup = Setup()

    setup.show()

    app.exec()
