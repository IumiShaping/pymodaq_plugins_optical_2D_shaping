from pymodaq_plugins_optical_2D_shaping.setup.factory import ElementFactory
from pymodaq_gui.parameter import Parameter



factory = ElementFactory()
print(factory.get_elements())

def test_factory_and_parameter():
    for index, elt_name in enumerate(factory.get_elements()):
        elt_obj = factory.create_element(elt_name, index=index)
        print(elt_obj)

        param = Parameter.create(name='param', type='group', children=elt_obj.to_list())

        assert elt_obj == elt_obj.from_parameter(param)