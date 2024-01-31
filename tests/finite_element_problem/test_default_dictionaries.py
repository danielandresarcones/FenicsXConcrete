import copy

import pytest

from fenicsxconcrete.finite_element_problem.base_material import MaterialProblem
from fenicsxconcrete.finite_element_problem.concrete_am import ConcreteAM
from fenicsxconcrete.finite_element_problem.concrete_thermo_mechanical import ConcreteThermoMechanical
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity
from fenicsxconcrete.util import ureg


# @pytest.mark.parametrize("material_model", [LinearElasticity, ConcreteAM, ConcreteThermoMechanical])
# def test_default_dictionaries(material_model: MaterialProblem) -> None:
#     """This function creates experimental setups with the respective default dictionaries
#
#     This makes sure all relevant values are included"""
#
#     default_setup, default_parameters = material_model.default_parameters()
#
#     fem_problem = material_model(default_setup, default_parameters)
#     fem_problem.solve()
#
#     # test that each parameter is truly required
#     # a loop over all default parameters removes each on in turn and expects a key error from the initialized problem
#     for key in default_parameters:
#         with pytest.raises(KeyError) as ex:
#             less_parameters = copy.deepcopy(default_parameters)
#             less_parameters.pop(key)
#             fem_problem = material_model(default_setup, less_parameters)
#             fem_problem.solve()
#             print(key, "seems to be an unneccessary key in the default dictionary")
#         print(ex)
# since the default parameter are used in each experiment or material problem this test makes no sence anymore
@pytest.mark.parametrize("material_model", [LinearElasticity, ConcreteAM, ConcreteThermoMechanical])
def test_dimensionality_check(material_model: MaterialProblem) -> None:

    default_setup, default_parameters = material_model.default_parameters()

    with pytest.raises(ValueError):
        default_parameters["g"] = 3 * ureg("m")  # gravity should be m/s²
        fem_problem = material_model(default_setup, default_parameters)


@pytest.mark.parametrize("material_model", [LinearElasticity, ConcreteAM, ConcreteThermoMechanical])
def test_default_parameters(material_model: MaterialProblem) -> None:
    """This function tests if the default_parameters are complete"""
    # default_material = LinearElasticity

    default_setup, default_parameters = material_model.default_parameters()

    try:
        fem_problem = material_model(default_setup, default_parameters)
        fem_problem.solve()
    except KeyError:
        print("default parameter dictionary is wrong")
        raise ValueError
