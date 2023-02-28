import re
from typing import Tuple, Optional

import gdb  # pylint: disable=E0401
import gdb.types  # pylint: disable=E0401
import numpy as np

from .type_handler import TypeHandler, ScalarTypeHandler

COMPLEX_REGEX = re.compile("(\\S*) \\+ (\\S*) \\* I")

class ArmadilloColVec(TypeHandler):
    @staticmethod
    def can_handle(gdb_type: gdb.Type) -> bool:
        return str(gdb_type).startswith("arma::Col") and str(gdb_type.template_argument(0)) != "bool"

    def shape(self, gdb_value: gdb.Value) -> Tuple[Optional[int], ...]:
        size = int(gdb_value["n_elem"])
        return (size,)

    def contained_type(self, gdb_value: gdb.Value) -> gdb.Type:
        return gdb_value.type.template_argument(0)

    def extract(self, gdb_value: gdb.Value, index: Tuple[int, ...]):
        return (gdb_value["mem"] + index[0]).dereference()

class ArmadilloRowVec(TypeHandler):
    @staticmethod
    def can_handle(gdb_type: gdb.Type) -> bool:
        return str(gdb_type).startswith("arma::Row") and str(gdb_type.template_argument(0)) != "bool"

    def shape(self, gdb_value: gdb.Value) -> Tuple[Optional[int], ...]:
        size = int(gdb_value["n_elem"])
        return (size,)

    def contained_type(self, gdb_value: gdb.Value) -> gdb.Type:
        return gdb_value.type.template_argument(0)

    def extract(self, gdb_value: gdb.Value, index: Tuple[int, ...]):
        return (gdb_value["mem"] + index[0]).dereference()

class ArmadilloMat(TypeHandler):
    @staticmethod
    def can_handle(gdb_type: gdb.Type) -> bool:
        return str(gdb_type).startswith("arma::Mat")

    def shape(self, gdb_value: gdb.Value) -> Tuple[Optional[int], ...]:
        n_cols = int(gdb_value["n_cols"])
        n_rows = int(gdb_value["n_rows"])
        return (n_rows, n_cols)

    def contained_type(self, gdb_value: gdb.Value) -> gdb.Type:
        return gdb_value.type.template_argument(0)

    def extract(self, gdb_value: gdb.Value, index: Tuple[int, ...]):
        n_rows = int(gdb_value["n_rows"])
        return (gdb_value["mem"] + index[1] * n_rows + index[0]).dereference()
