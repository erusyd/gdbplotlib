from .type_set import TypeSet
from . import std_types
from . import arma_types

default = TypeSet()
default.register(std_types.StdVector)
default.register(std_types.StdVectorBool)
default.register(std_types.StdArray)
default.register(std_types.Pointer)
default.register(std_types.Array)
default.register(std_types.Double)
default.register(std_types.Float)
default.register(std_types.StdComplexDouble)
default.register(std_types.StdComplexFloat)
default.register(std_types.Integral)
default.register(std_types.Bool)
default.register(arma_types.ArmadilloColVec)
default.register(arma_types.ArmadilloRowVec)
default.register(arma_types.ArmadilloMat)
