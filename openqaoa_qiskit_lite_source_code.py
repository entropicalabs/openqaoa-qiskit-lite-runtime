from typing import List, Union, Tuple, Any, Callable, Iterable, Optional, Dict, Type
from collections import Counter, defaultdict
from sympy import Symbol
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod, abstractproperty
import json
import scipy
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import deepcopy
import time
import math

# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# from qiskit_ibm_provider.job.exceptions import (
#     IBMJobApiError,
#     IBMJobInvalidStateError,
#     IBMJobFailureError,
#     IBMJobTimeoutError,
# )
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    XGate,
    RXGate,
    RYGate,
    RZGate,
    CXGate,
    CZGate,
    RXXGate,
    RZXGate,
    RZZGate,
    RYYGate,
    CRZGate,
)
# from qiskit_ibm_provider import IBMProvider
# from qiskit_aer import AerSimulator

# import pickle

from scipy.optimize._minimize import minimize, MINIMIZE_METHODS
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds


class Publisher:
    """Class used to publish interim results."""

    def __init__(self, messenger):
        self._messenger = messenger

    def callback(self, *args, **kwargs):
        text = list(args)
        for k, v in kwargs.items():
            text.append({k: v})
        self._messenger.publish(text)


# UTILITY REQUIREMENTS FOR PROBLEMS ###################


def check_kwargs(list_expected_params, list_default_values, **kwargs):
    """
    Checks that the given list of expected parameters can be found in the
    kwargs given as input. If so, it returns the parameters from kwargs, else
    it raises an exception.

    Args:
        list_expected_params: List[str]
            List of string containing the name of the expected parameters in
            kwargs
        list_default_values: List
            List containing the deafult values of the expected parameters in
            kwargs
        **kwargs:
            Keyword arguments where keys are supposed to be the expected params

    Returns:
        A tuple with the actual expected parameters if they are found in kwargs.

    Raises:
        ValueError:
            If one of the expected arguments is not found in kwargs and its
            default value is not specified.
    """

    def check_kwarg(expected_param, default_value, **kwargs):
        param = kwargs.pop(expected_param, default_value)

        if param is None:
            raise ValueError(f"Parameter '{expected_param}' should be specified")

        return param

    params = []
    for expected_param, default_value in zip(list_expected_params, list_default_values):
        params.append(check_kwarg(expected_param, default_value, **kwargs))

    return tuple(params)


def delete_keys_from_dict(obj: Union[list, dict], keys_to_delete: List[str]):
    """
    Recursively delete all the keys keys_to_delete from a object (or list of dictionaries)
    Parameters
    ----------
    obj: dict or list[dict]
        dictionary or list of dictionaries from which we want to delete keys
    keys_to_delete: list
        list of keys to delete from the dictionaries

    Returns
    -------
    obj: dict or list[dict]
        dictionary or list of dictionaries from which we have deleted keys
    """
    if isinstance(obj, dict):
        for key in keys_to_delete:
            if key in obj:
                del obj[key]
        for key in obj:
            if isinstance(obj[key], dict):
                delete_keys_from_dict(obj[key], keys_to_delete)
            elif isinstance(obj[key], list):
                for item in obj[key]:
                    delete_keys_from_dict(item, keys_to_delete)
    elif isinstance(obj, list):
        for item in obj:
            delete_keys_from_dict(item, keys_to_delete)

    return obj


def convert2serialize(obj, complex_to_string: bool = False):
    """
    Recursively converts object to dictionary.

    Parameters
    ----------
    obj: object
        Object to convert to dictionary.
    complex_to_string: bool
        If True, convert complex numbers to string, so the result can be serialized to JSON.

    Returns
    -------
    dict: dict
        Dictionary representation of the object.
    """
    if isinstance(obj, dict):
        return {
            k: convert2serialize(v, complex_to_string)
            for k, v in obj.items()
            if v is not None
        }
    elif hasattr(obj, "_ast"):
        return convert2serialize(obj._ast(), complex_to_string)
    elif isinstance(obj, tuple):
        return tuple(
            convert2serialize(v, complex_to_string) for v in obj if v is not None
        )
    elif not isinstance(obj, str) and hasattr(obj, "__iter__"):
        return [convert2serialize(v, complex_to_string) for v in obj if v is not None]
    elif hasattr(obj, "__dict__"):
        return {
            k: convert2serialize(v, complex_to_string)
            for k, v in obj.__dict__.items()
            if not callable(v) and v is not None
        }
    elif complex_to_string and isinstance(obj, complex):
        return str(obj)
    else:
        return obj


def round_value(function):
    """
    Round a value to a given precision.
    This function will be used as a decorator to round the values given by the
    ``expectation`` and ``expectation_w_uncertainty`` methods.

    Parameters
    ----------
    function: `Callable`
        The function to be decorated

    Returns
    -------
        The rounded value(s)

    """

    PRECISION = 12

    def wrapper(*args, **kwargs):
        values = function(*args, **kwargs)
        if isinstance(values, dict):
            return {k: round(v, PRECISION) for k, v in values.items()}
        else:
            return np.round(values, PRECISION)

    return wrapper


def bitstring_energy(hamiltonian, bitstring: Union[List[int], str]) -> float:
    """
    Computes the energy of a given bitstring with respect to a classical cost Hamiltonian.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object determining the energy levels.
    bitstring : `list` or `str`
        A list of integers 0 and 1, or a string, representing a configuration.

    Returns
    -------
    energy: `float`
        The energy of the given bitstring with respect to the cost Hamiltonian.
    """
    # Initialize energy value
    energy = 0

    # Compute energy contribution term by term
    for i, term in enumerate(hamiltonian.terms):
        # Compute sign of spin interaction term
        variables_product = np.prod(
            [(-1) ** int(bitstring[k]) for k in term.qubit_indices]
        )

        # Add energy contribution
        energy += hamiltonian.coeffs[i] * variables_product

    # Add constant contribution
    energy += hamiltonian.constant

    return energy


def flip_counts(counts_dictionary: dict) -> dict:
    """
    Returns a counts/probability dictionary that have their keys flipped. This
    formats the bit-strings from a right-most bit representing being the first
    qubit to the left-most bit representing the first qubit.

    Parameters
    ----------
    counts_dictionary: `dict`
        Count dictionary whose keys are flipped.

    Returns
    -------
    output_counts_dictionary: `dict`
        Count dictionary with flipped keys.
    """

    output_counts_dictionary = dict()

    for key, value in counts_dictionary.items():
        output_counts_dictionary[key[::-1]] = value

    return output_counts_dictionary


# Hamiltonian, operators required for OQ ###################################

Identity = np.array(([1, 0], [0, 1]), dtype=complex)
PauliX = np.array(([0, 1], [1, 0]), dtype=complex)
PauliY = np.array(([0, -1j], [1j, 0]), dtype=complex)
PauliZ = np.array(([1, 0], [0, -1]), dtype=complex)

PAULIS_SET = set("XYZI")

PAULI_MULT_RULES = {
    "XX": "I",
    "YY": "I",
    "ZZ": "I",
    "XY": "Z",
    "YX": "Z",
    "XZ": "Y",
    "ZX": "Y",
    "YZ": "X",
    "ZY": "X",
}
PAULI_MULT_RULES.update({f"I{op}": op for op in PAULIS_SET})
PAULI_MULT_RULES.update({f"{op}I": op for op in PAULIS_SET})

PAULI_MAPPERS = {
    "X": PauliX,
    "Y": PauliY,
    "Z": PauliZ,
    "I": Identity,
    "iX": 1j * PauliX,
    "iY": 1j * PauliY,
    "iZ": 1j * PauliZ,
    "-iX": -1j * PauliX,
    "-iY": -1j * PauliY,
    "-iZ": -1j * PauliZ,
}

PAULI_PHASE_MAPPERS = {
    "XX": 1,
    "YY": 1,
    "ZZ": 1,
    "XY": 1j,
    "ZX": 1j,
    "YZ": 1j,
    "ZY": -1j,
    "XZ": -1j,
    "YX": -1j,
    "X": 1,
    "Y": 1,
    "Z": 1,
    "I": 1,
}
PAULI_PHASE_MAPPERS.update({f"I{op}": 1 for op in PAULIS_SET})
PAULI_PHASE_MAPPERS.update({f"{op}I": 1 for op in PAULIS_SET})


class PauliOp:
    """
    Pauli operator class to handle Pauli operators.

    Attributes
    ----------
    qubit_indices: `Tuple[int]`
        Indices of each spin in the Pauli operators.

    pauli_str: `str`
        Names of each Pauli opeator acting on each spin.

    phase: `complex`
        Overall complex phase of the Pauli operators.

    is_trivial

    matrix
    """

    def __init__(self, pauli_str: str, qubit_indices: Tuple[int]):
        """
        Initialize the Pauli Operator object.

        Parameters
        ----------
        pauli_str: `str`
                The Pauli operator basis string.
        qubit_indices: `Tuple[int]`
                The qubits on which the Pauli operates.

        Attributes
        ----------
        pauli_str: `str`
                The Pauli operator basis string.
        qubit_indices: `Tuple[int]`
                The qubits on which the Pauli operates.
        phase: `complex`
                The phase of the Pauli operator.
        """
        # Ensure number of indices matches number of declared operator names
        assert len(pauli_str) == len(
            qubit_indices
        ), "Each Pauli operator must have a unique qubit index"

        # Simplify Pauli string if multiple operators act on the same qubit
        pauli_str, qubit_indices, phase = self._simplify(pauli_str, qubit_indices)

        # Sort indices and strings
        self.qubit_indices, self.pauli_str = self._sort_pauli_op(
            qubit_indices, pauli_str
        )

        # Store phase accumulated from simplification
        self.phase = phase

    def __hash__(self) -> int:
        return hash((self.qubit_indices, self.pauli_str, self.phase))

    @staticmethod
    def _sort_pauli_op(qubit_indices: Tuple[int], pauli_str: str):
        """
        Sort the Pauli Operator in increasing order of qubit indices.

        Parameters
        ----------
        qubit_indices: `Tuple[int]`
                The qubit indices of the Pauli Operator.
        pauli_str: `str`
                The Pauli Operator basis string.

        Returns
        -------
        sorted_qubit_indices: `Tuple[int]`
                The sorted qubit indices in increasing order.
        sorted_pauli_str: `str`
                The sorted Pauli Operator basis string.

        Examples
        --------
        >>> PauliOp('YZX',(3,1,2)) -> PauliOp('ZXY',(1,2,3)) with appropriate phase.
        """
        # Initialize sorted pauli string and indices
        sorted_pauli_str = ""
        sorted_qubit_indices = []

        # Sorting
        for index, string in sorted(zip(qubit_indices, pauli_str)):
            # Ensure string is a valid Pauli operator
            if string not in PAULIS_SET:
                raise ValueError(
                    f"{string} is not a valid Pauli. Please choose from the set {PAULIS_SET}"
                )

            # Store sorted indices and strings
            sorted_qubit_indices.append(index)
            sorted_pauli_str += string

        sorted_qubit_indices = tuple(sorted_qubit_indices)

        return sorted_qubit_indices, sorted_pauli_str

    @staticmethod
    def _simplify(pauli_str: str, qubit_indices: Tuple[int]):
        """
        Simplify the definition of Pauli Operator.

        Parameters
        ----------
        qubit_indices: `Tuple[int]`
                The qubit indices of the Pauli Operator.
        pauli_str: `str`
                The Pauli Operator basis string.

        Returns
        -------
        new_pauli_str: `str`
                The updated Pauli Operator basis string.
        new_qubit_indices: `Tuple[int]`
                The updated qubit indices in increasing order.

        Examples
        --------
        PauliOp('XZX',(3,2,2)) -> PauliOp('XY',(3,2)) with appropriate phase.
        """
        new_phase = 1

        qubit_reps = Counter(qubit_indices)

        # If no repetitions, do nothing
        if len(qubit_indices) == len(qubit_reps.values()):
            new_pauli_str = pauli_str
            new_qubit_indices = qubit_indices

        # If repetitions present, simplify operator
        else:
            # Extract spins with multiple operators
            repeating_indices = [index for index, rep in qubit_reps.items() if rep > 1]

            paulis_list_to_contract = []

            # Extract operators to be simpilfied
            for index in repeating_indices:
                paulis_list_to_contract.append(
                    [
                        pauli_str[i]
                        for i in range(len(qubit_indices))
                        if qubit_indices[i] == index
                    ]
                )

            # Simplify
            for paulis in paulis_list_to_contract:
                i = 0

                # Repeat until all operators have been simplified
                while len(paulis) > 1:
                    pauli_mult = paulis[i] + paulis[i + 1]
                    paulis[0] = PAULI_MULT_RULES[pauli_mult]
                    new_phase *= PAULI_PHASE_MAPPERS[pauli_mult]
                    paulis.pop(i + 1)

            # Store simplified strings and indices
            repeating_pauli_str = "".join(pauli[0] for pauli in paulis_list_to_contract)
            non_repeating_indices = [
                index for index, rep in qubit_reps.items() if rep == 1
            ]
            non_repeating_paulis = "".join(
                pauli_str[list(qubit_indices).index(idx)]
                for idx in non_repeating_indices
            )

            new_pauli_str = non_repeating_paulis + repeating_pauli_str
            new_qubit_indices = tuple(non_repeating_indices + repeating_indices)

        return new_pauli_str, new_qubit_indices, new_phase

    @property
    def _is_trivial(self) -> bool:
        """
        Returns `True` if the PauliOp only contains identity terms `'I'`.
        """
        return self.pauli_str == "I" * len(self.qubit_indices)

    @property
    def matrix(self):
        """
        Matrix representation of the Pauli Operator.
        """
        # Initialize matrix representation
        mat = PAULI_MAPPERS[self.pauli_str[0]]

        for pauli in self.pauli_str[1:]:
            mat = np.kron(mat, PAULI_MAPPERS[pauli])
        return mat

    def __len__(self):
        """
        Length of the Pauli term.
        """
        return len(self.qubit_indices)

    def __eq__(self, other_pauli_op):
        """
        Check whether two pauli_operators are equivalent, by comparing qubit indices and pauli strings.
        """
        tuple_1 = (self.qubit_indices, self.pauli_str, self.phase)
        tuple_2 = (
            other_pauli_op.qubit_indices,
            other_pauli_op.pauli_str,
            other_pauli_op.phase,
        )
        return tuple_1 == tuple_2

    def __copy__(self):
        """
        Create a new `PauliOp` by copying the current one.
        """
        copied_pauli_op = self.__class__.__new__(self.__class__)
        for attribute, value in vars(self).items():
            setattr(copied_pauli_op, attribute, value)
        return copied_pauli_op

    def __str__(self):
        """
        String representation of the Pauli Operator.
        """
        term_str = "".join(
            pauli_base + "_" + str({index})
            for pauli_base, index in zip(self.pauli_str, self.qubit_indices)
        )
        return term_str

    def __repr__(self):
        """
        String representation of the Pauli Operator object.
        """
        term_repr = f"PauliOp({self.pauli_str},{self.qubit_indices})"
        return term_repr

    def __mul__(self, other_pauli_op):
        """
        Multiply two Pauli Operators.

        Parameters
        ----------
        other_pauli_op: `PauliOp`
                The other Pauli Operator to be multiplied.

        Return
        ------
        new_pauli_op: `PauliOp`
                The resulting Pauli Operator after the multiplication.
        """
        assert isinstance(other_pauli_op, PauliOp), "Please specify a Pauli Operator"

        copied_current_pauli_op = self.__copy__()
        copied_current_pauli_op.__matmul__(other_pauli_op)

        return copied_current_pauli_op

    def __matmul__(self, other_pauli_op):
        """
        In-place Multiplication of Pauli Operators. Contract `other_pauli_op` into `self`.

        Parameters
        ----------
        other_pauli_op: `PauliOp`
                The Pauli Operator object to be multiplied.
        """
        assert isinstance(other_pauli_op, PauliOp), "Please specify a Pauli Operator"

        # Combined number of qubits
        n_qubits = max(max(self.qubit_indices), max(other_pauli_op.qubit_indices)) + 1

        self_pauli_str_list = list(self.pauli_str)
        other_pauli_str_list = list(other_pauli_op.pauli_str)

        # Fill pauli strings with identity operators for Pauli objects
        # for the number of strings to match the combined number of qubits
        for i in range(n_qubits):
            if i not in self.qubit_indices:
                self_pauli_str_list.insert(i, "I")
            if i not in other_pauli_op.qubit_indices:
                other_pauli_str_list.insert(i, "I")

        new_full_operator = ""
        new_phase = 1

        # Perform multiplication
        for idx in range(n_qubits):
            pauli_composition = self_pauli_str_list[idx] + other_pauli_str_list[idx]
            mult_phase = PAULI_PHASE_MAPPERS[pauli_composition]
            mult_pauli = PAULI_MULT_RULES[pauli_composition]

            new_full_operator += mult_pauli
            new_phase *= mult_phase

        self.qubit_indices = tuple(
            [idx for idx in range(n_qubits) if new_full_operator[idx] != "I"]
        )
        self.pauli_str = new_full_operator.replace("I", "")
        self.phase = new_phase

        return self

    @classmethod
    def X(cls, qubit_idx):
        """
        Pauli X operator.
        """
        return cls("X", (qubit_idx,))

    @classmethod
    def Y(cls, qubit_idx):
        """
        Pauli Y operator.
        """
        return cls("Y", (qubit_idx,))

    @classmethod
    def Z(cls, qubit_idx):
        """
        Pauli Z operator.
        """
        return cls("Z", (qubit_idx,))

    @classmethod
    def I(cls, qubit_idx):
        """
        Pauli identity operator.
        """
        return cls("I", (qubit_idx,))


class Hamiltonian:
    """
    General Hamiltonian class.

    Attributes
    ----------
    n_qubits: `int`
    terms: `List[PauliOp]`
    coeffs: `List[complex,float,int]`
    constant: `float`
    qubits_pairs: `List[PauliOp]`
    qubits_singles: `List[PauliOp]`
    single_qubit_coeffs: `List[float]`
    pair_qubit_coeffs: `List[float]`
    qureg
    expression
    hamiltonian_squared
    """

    def __init__(
        self,
        pauli_terms: List[PauliOp],
        coeffs: List[Union[complex, int, float]],
        constant: float,
        divide_into_singles_and_pairs: bool = True,
    ):
        """
        Parameters
        ----------
        pauli_terms: `List[PauliOp]`
            Set of terms in the Hamiltonian as PauliOp objects.
        coeffs: `List[Union[complex,float,int]]`
            Multiplicative coefficients for each Pauli term in the Hamiltonian.
        constant: `float`
            Constant term in the Hamiltonian.
        divide_into_singles_and_pairs: `bool`, optional
                Whether to divide the Hamiltonian into singles and pairs
        """
        assert len(pauli_terms) == len(
            coeffs
        ), "Number of Pauli terms in Hamiltonian should be same as number of coefficients"

        physical_qureg = []

        # Extract physical regiser from qubit indices
        for pauli_term in pauli_terms:
            if isinstance(pauli_term, PauliOp):
                physical_qureg.extend(pauli_term.qubit_indices)
            else:
                raise TypeError(
                    f"Pauli terms should be of type PauliOp and not {type(pauli_term)}"
                )
        physical_qureg = list(set(physical_qureg))

        # Number of qubits
        self.n_qubits = len(physical_qureg)

        # Extract qubit map if necessary
        need_remapping = False
        if physical_qureg != self.qureg:
            print(
                f"Qubits in the specified Hamiltonian are remapped to {self.qureg}."
                "Please specify the physical quantum register as a qubit layout argument in the backend"
            )
            need_remapping = True
            qubit_mapper = dict(zip(physical_qureg, self.qureg))

        self.terms = []
        self.coeffs = []
        self.constant = constant

        for term, coeff in zip(pauli_terms, coeffs):
            # Identity terms are added to the constant
            if term._is_trivial:
                self.constant += coeff

            # Remap terms if required
            else:
                if need_remapping:
                    new_indices = tuple(qubit_mapper[i] for i in term.qubit_indices)
                    pauli_str = term.pauli_str
                    self.terms.append(PauliOp(pauli_str, new_indices))
                else:
                    self.terms.append(term)

                # Update the coefficients with phase from Pauli Operators
                self.coeffs.append(coeff * term.phase)
                # after absorbing the phase in coeff, set the phase in term to 1
                term.phase = 1

        if divide_into_singles_and_pairs:
            self._divide_into_singles_pairs()

    @property
    def qureg(self):
        """
        List of qubits from 0 to n-1 in Hamiltonian.
        """
        return list(range(self.n_qubits))

    def _divide_into_singles_pairs(self):
        """
        Extract terms and coefficients for linear and quadratic terms in the Hamiltonian.
        """

        self.qubits_pairs = []
        self.qubits_singles = []
        self.single_qubit_coeffs = []
        self.pair_qubit_coeffs = []

        for term, coeff in zip(self.terms, self.coeffs):
            if len(term) == 1:
                self.qubits_singles.append(term)
                self.single_qubit_coeffs.append(coeff)
            elif len(term) == 2:
                self.qubits_pairs.append(term)
                self.pair_qubit_coeffs.append(coeff)
            else:
                raise NotImplementedError(
                    "Hamiltonian only supports Linear and Quadratic terms"
                )

    def __str__(self):
        """
        Return a string representation of the Hamiltonian.
        """
        hamil_str = ""
        for coeff, term in zip(self.coeffs, self.terms):
            hamil_str += str(np.round(coeff, 3)) + "*" + term.__str__() + " + "
        hamil_str += str(self.constant)
        return hamil_str

    def __len__(self):
        """
        Return the number of terms in the Hamiltonian.
        """
        return len(self.terms)

    @property
    def expression(self):
        """
        Generates a symbolic expression for the Hamiltonian.

        Returns
        -------
        hamiltonian_expression: `sympy.Symbol`
            Symbolic expression for the Hamiltonian.
        """

        # Generate expression
        hamiltonian_expression = Symbol(str(self.constant))
        for term, coeff in zip(self.terms, self.coeffs):
            hamiltonian_expression += Symbol(str(coeff) + term.__str__())
        return hamiltonian_expression

    # A function that outputs a dictionary of the Hamiltonian terms and coefficients
    def hamiltonian_dict(self, classical: bool = True):
        """
        Generates a dictionary of the Hamiltonian terms and coefficients.

        Parameters
        ----------
        classical: `bool`, optional
            If true, returns a dictionary containing only qubit indices as terms
            else returns a dictionary containing PauliOp objects as terms.

        Returns
        -------
        `dict`
            Dictionary of the Hamiltonian terms and coefficients.
        """
        hamiltonian_dict = {}
        for term, coeff in zip(self.terms, self.coeffs):
            if classical:
                hamiltonian_dict[tuple(term.qubit_indices)] = coeff
            else:
                hamiltonian_dict[term] = coeff
            # add the constant term
            hamiltonian_dict[()] = self.constant
        return hamiltonian_dict

    def __add__(self, other_hamiltonian):
        """
        Add two Hamiltonians in place updating `self`

        Parameters
        ----------
        other_hamiltonian: `Hamiltonian`
                The other Hamiltonian to be added
        """
        assert isinstance(other_hamiltonian, Hamiltonian)

        for other_term, other_coeff in zip(
            other_hamiltonian.terms, other_hamiltonian.coeffs
        ):
            if other_term in self.terms:
                self.coeffs[self.terms.index(other_term)] += other_coeff
            else:
                self.terms.append(other_term)
                self.coeffs.append(other_coeff)

    @property
    def hamiltonian_squared(self):
        """
        Compute the squared of the Hamiltonian, necessary for computing
        the error in expectation values.

        Returns
        -------
        hamil_squared: `Hamiltonian`
            Hamiltonian squared.
        """
        hamil_sq_terms = []
        hamil_sq_coeffs = []
        hamil_sq_constant = self.constant**2

        for i, term1 in enumerate(self.terms):
            for j, term2 in enumerate(self.terms):
                new_term = term1 * term2

                # If multiplication yields a constant add it to hamiltonian constant
                if new_term.pauli_str == "":
                    hamil_sq_constant += self.coeffs[i] * self.coeffs[j]

                # If it yields non-trivial term add it to list of terms
                else:
                    new_phase = new_term.phase
                    new_coeff = self.coeffs[i] * self.coeffs[j]
                    hamil_sq_terms.append(new_term)
                    hamil_sq_coeffs.append(new_coeff)

        hamil_sq_terms.extend(self.terms)
        hamil_sq_coeffs.extend([2 * self.constant * coeff for coeff in self.coeffs])

        hamil_squared = Hamiltonian(
            hamil_sq_terms,
            hamil_sq_coeffs,
            hamil_sq_constant,
            divide_into_singles_and_pairs=False,
        )
        return hamil_squared

    @classmethod
    def classical_hamiltonian(
        cls,
        terms: List[Union[Tuple, List]],
        coeffs: List[Union[float, int]],
        constant: float,
    ):
        """
        Generates a classical Hamiltonian from a list of terms, coefficients and constant.

        Parameters
        ----------
        terms: `List[tuple]` or `List[list]`
            Set of qubit indices for each term in the Hamiltonian.
        coeffs: `List[float]` or `List[int]`
            Coefficients associated with each term in the Hamiltonian
        constant: `float`
            Constant term in the Hamiltonian.

        Returns
        -------
        Hamiltonian:
            Classical Hamiltonian.
        """
        pauli_ops = []
        pauli_coeffs = []

        for term, coeff in zip(terms, coeffs):
            # Check coeffcient type
            if not isinstance(coeff, int) and not isinstance(coeff, float):
                raise ValueError(
                    "Classical Hamiltonians only support Integer or Float coefficients"
                )

            # Construct Hamiltonian terms from Pauli operators
            if len(term) == 2:
                pauli_ops.append(PauliOp("ZZ", term))
                pauli_coeffs.append(coeff)
            elif len(term) == 1:
                pauli_ops.append(PauliOp("Z", term))
                pauli_coeffs.append(coeff)
            elif len(term) == 0:
                constant += coeff
            else:
                raise ValueError("Hamiltonian only supports Linear and Quadratic terms")

        return cls(pauli_ops, pauli_coeffs, constant)


# UTILITY FUNCTION TO CREATE X-MIXER HAMILTONIAN


def X_mixer_hamiltonian(n_qubits: int, coeffs: List[float] = None) -> Hamiltonian:
    """Construct a Hamiltonian object to implement the X mixer.

    Parameters
    ----------
    n_qubits: `int`
        The number of qubits in the mixer Hamiltonian.
    coeffs: `List[float]`
        The coefficients of the X terms in the Hamiltonian.

    Returns
    -------
    `Hamiltonian`
        The Hamiltonian object corresponding to the X mixer.
    """
    # If no coefficients provided, set all to -1
    coeffs = [-1] * n_qubits if coeffs is None else coeffs

    # Initialize list of terms
    terms = []

    # Generate terms in the X mixer
    for i in range(n_qubits):
        terms.append(PauliOp.X(i))

    # Define mixer Hamiltonian
    hamiltonian = Hamiltonian(pauli_terms=terms, coeffs=coeffs, constant=0)

    return hamiltonian


# DEFINE THE PROBLEM FOR QAOA ########################


class QUBO(object):
    """
    Creates an instance of Quadratic Unconstrained Binary Optimization (QUBO)
    class, which offers a way to encode optimization problems.

    Parameters
    ----------
    n: int
        The number of variables in the representation.
    terms: List[Tuple[int, ...],List]
        The different terms in the QUBO encoding, indicating the
        different interactions between variables.
    weights: List[float]
        The list of weights (or coefficients) corresponding to each
        interaction defined in `terms`.
    clean_terms_and_weights: bool
        Boolean indicating whether terms and weights can be cleaned


    Returns
    -------
        An instance of the Quadratic Unconstrained Binary Optimization (QUBO) class.
    """

    # Maximum number of terms allowed to enable the cleaning procedure
    TERMS_CLEANING_LIMIT = 5000

    def __init__(
        self,
        n,
        terms,
        weights,
        problem_instance: dict = {"problem_type": "generic_qubo"},
        clean_terms_and_weights=False,
    ):
        # check-type for terms and weights
        if not isinstance(terms, list) and not isinstance(terms, tuple):
            raise TypeError(
                "The input parameter terms must be of type of list or tuple"
            )

        if not isinstance(weights, list) and not isinstance(weights, tuple):
            raise TypeError(
                "The input parameter weights must be of type of list or tuple"
            )

        for each_entry in weights:
            if not isinstance(each_entry, float) and not isinstance(each_entry, int):
                raise TypeError(
                    "The elements in weights list must be of type float or int."
                )

        terms = list(terms)
        weights = list(weights)

        # Check that terms and weights have matching lengths
        if len(terms) != len(weights):
            raise ValueError("The number of terms and number of weights do not match")

        constant = 0
        try:
            constant_index = [i for i, term in enumerate(terms) if len(term) == 0][0]
            constant = weights.pop(constant_index)
            terms.pop(constant_index)
        except:
            pass

        # If the user wants to clean the terms and weights or if the number of
        # terms is not too big, we go through the cleaning process
        if clean_terms_and_weights or len(terms) <= QUBO.TERMS_CLEANING_LIMIT:
            self.terms, self.weights = QUBO.clean_terms_and_weights(terms, weights)
        else:
            self.terms, self.weights = terms, weights

        self.constant = constant
        self.n = n

        # attribute to store the problem instance, it will be checked
        # if it is json serializable in the __setattr__ method
        self.problem_instance = problem_instance

        # Initialize the metadata dictionary
        self.metadata = {}

    def __iter__(self):
        for key, value in self.__dict__.items():
            # remove "_" from the beginning of the key if it exists
            yield (key[1:] if key.startswith("_") else key, value)

    def __setattr__(self, __name, __value):
        # check if problem_instance is json serializable, also check if
        # metadata is json serializable
        if __name == "problem_instance" or __name == "metadata":
            try:
                _ = json.dumps(__value)
            except Exception as e:
                raise e

        super().__setattr__(__name, __value)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, input_n):
        if not isinstance(input_n, int):
            raise TypeError("The input parameter, n, has to be of type int")

        if input_n <= 0:
            raise TypeError(
                "The input parameter, n, must be a positive integer greater than 0"
            )

        self._n = input_n

    def set_metadata(self, metadata: dict = {}):
        """


        Parameters
        ----------
        metadata: dict
            The metadata of the problem. All keys and values will
            be stored in the metadata dictionary.
        """

        # update the metadata (it will be checked if it is json
        # serializable in the __setattr__ method)
        self.metadata = {**self.metadata, **metadata}

    def asdict(self, exclude_keys: List[str] = []):
        """
        Returns a dictionary containing the serialization of the class.

        Parameters
        ----------
        exclude_keys: List[str]


        Returns
        -------
            A dictionary containing the serialization of the class.
        """

        if exclude_keys == []:
            return convert2serialize(dict(self))
        else:
            return delete_keys_from_dict(
                obj=convert2serialize(dict(self)), keys_to_delete=exclude_keys
            )

    @staticmethod
    def from_dict(dict: dict, clean_terms_and_weights=False):
        """


        Parameters
        ----------
        dict: dict
            The dictionary containing the serialization of the QUBO object.
        clean_terms_and_weights: bool


        Returns
        -------
            A QUBO object.
        """

        # make a copy of the dictionary to avoid modifying the input
        dict = dict.copy()

        # extract the metadata
        metadata = dict.pop("metadata", {})

        # make a copy of the terms and weights to avoid modifying the input
        dict["terms"] = dict["terms"].copy()
        dict["weights"] = dict["weights"].copy()

        # add the constant term
        dict["terms"].append([])
        dict["weights"].append(dict.pop("constant", 0))

        # create the QUBO object
        qubo = QUBO(**dict, clean_terms_and_weights=clean_terms_and_weights)

        # add the metadata
        qubo.metadata = metadata.copy()

        # return the QUBO object
        return qubo

    @staticmethod
    def clean_terms_and_weights(terms, weights):
        """Goes through the terms and weights and group them when possible"""
        # List to record the terms as sets
        unique_terms = []

        # Will record the weight for the unique terms (note that since Sets are
        # unhashable in Python, we use a dict with integers for the keys, that
        # are mapped with the corresponding indices of terms from unique_terms)
        new_weights_for_terms = defaultdict(float)

        # We do one pass over terms and weights
        for term, weight in zip(terms, weights):
            # Convert the term to a set
            term_set = set(term)

            # If this term is not yet recorded, we add it to the list of unique
            # terms and we use that it is the last element to find its index
            if term_set not in unique_terms:
                unique_terms.append(term_set)
                term_index = len(unique_terms) - 1

            # Else if the term is alreaddy recorded, we just need to retrieve
            # its index in the unique_terms list
            else:
                term_index = unique_terms.index(term_set)

            # Update the weight in the dictionary using the retrieved index
            new_weights_for_terms[term_index] += weight

        # Return terms and weights, making sure to convert the terms back to lists
        return (
            [list(term) for term in unique_terms],
            list(new_weights_for_terms.values()),
        )

    @staticmethod
    def random_instance(n, density=0.5, format_m="coo", max_abs_value=100):
        # Generate a random matrix (elements in [0, 1]) of type sparse
        random_matrix = scipy.sparse.rand(n, n, density=density, format=format_m)

        # Retrieve the indices of non-zero elements of the matrix as list of tuples
        terms = np.transpose(random_matrix.nonzero())

        # Get the matrix entries in a list, but scale the elements and
        # make them centered at 0 by subtracting 0.5
        weights = max_abs_value * (random_matrix.data - 0.5)

        # Return the terms and weights, taking care of converting to the correct types
        return QUBO(n, [list(map(int, i)) for i in terms], [float(i) for i in weights])

    @staticmethod
    def convert_qubo_to_ising(n, qubo_terms, qubo_weights):
        """Convert QUBO terms and weights to their Ising equivalent"""
        ising_terms, ising_weights = [], []
        constant_term = 0
        linear_terms = np.zeros(n)

        # Process the given terms and weights
        for term, weight in zip(qubo_terms, qubo_weights):
            if len(term) == 2:
                u, v = term

                if u != v:
                    ising_terms.append([u, v])
                    ising_weights.append(weight / 4)
                else:
                    constant_term += weight / 4

                linear_terms[term[0]] -= weight / 4
                linear_terms[term[1]] -= weight / 4
                constant_term += weight / 4
            elif len(term) == 1:
                linear_terms[term[0]] -= weight / 2
                constant_term += weight / 2
            else:
                constant_term += weight

        for variable, linear_term in enumerate(linear_terms):
            ising_terms.append([variable])
            ising_weights.append(linear_term)

        ising_terms.append([])
        ising_weights.append(constant_term)
        return ising_terms, ising_weights

    @property
    def hamiltonian(self):
        """
        Returns the Hamiltonian of the problem.
        """
        return Hamiltonian.classical_hamiltonian(
            self.terms, self.weights, self.constant
        )


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the problem.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments

        Returns
        -------
            A random instance of the problem.
        """
        pass

    def __iter__(self):
        for key, value in self.__dict__.items():
            # remove "_" from the beginning of the key if it exists
            new_key = key[1:] if key.startswith("_") else key
            # convert networkx graphs to dictionaries for serialization
            # (to get back to a graph, use nx.node_link_graph)
            new_value = (
                nx.node_link_data(value) if isinstance(value, nx.Graph) else value
            )
            yield (new_key, new_value)

    @property
    def problem_instance(self):
        """
        Returns a dictionary containing the serialization of the class and
        the problem type name, which will be passed as metadata to the QUBO class.

        Returns
        -------
            A dictionary containing the serialization of the class
            and the problem type name.
        """
        return {**{"problem_type": self.__name__}, **dict(self)}


class MaximumCut(Problem):
    """
    Creates an instance of the Maximum Cut problem.

    Parameters
    ----------
    G: nx.Graph
        The input graph as NetworkX graph instance.

    Returns
    -------
        An instance of the Maximum Cut problem.
    """

    __name__ = "maximum_cut"

    DEFAULT_EDGE_WEIGHT = 1.0

    def __init__(self, G):
        self.G = G

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, input_networkx_graph):
        if not isinstance(input_networkx_graph, nx.Graph):
            raise TypeError("Input problem graph must be a networkx Graph.")

        # Relabel nodes to integers starting from 0
        mapping = dict(
            zip(input_networkx_graph, range(input_networkx_graph.number_of_nodes()))
        )
        self._G = nx.relabel_nodes(input_networkx_graph, mapping)

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Maximum Cut problem, whose graph is
        random following the Erdos-Renyi model.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments are:

            n_nodes: int
                The number of nodes (vertices) in the graph.
            edge_probability: float
                The probability with which an edge is added to the graph.

        Returns
        -------
            A random instance of the Maximum Cut problem.
        """
        n_nodes, edge_probability = check_kwargs(
            ["n_nodes", "edge_probability"], [None, None], **kwargs
        )
        seed = kwargs.get("seed", None)

        G = nx.generators.random_graphs.fast_gnp_random_graph(
            n=n_nodes, p=edge_probability, seed=seed
        )
        return MaximumCut(G)

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        # Iterate over edges (with weight) and store accordingly
        terms = []
        weights = []

        for u, v, edge_weight in self.G.edges(data="weight"):
            terms.append([u, v])

            # We expect the edge weight to be given in the attribute called
            # "weight". If it is None, assume a weight of 1.0
            weights.append(
                edge_weight if edge_weight else MaximumCut.DEFAULT_EDGE_WEIGHT
            )

        return QUBO(self.G.number_of_nodes(), terms, weights, self.problem_instance)


# BASEPARAMS #######################################


def _is_iterable_empty(in_iterable):
    if isinstance(in_iterable, Iterable):  # Is Iterable
        return all(map(_is_iterable_empty, in_iterable))
    return False  # Not an Iterable


class shapedArray:
    """Decorator-Descriptor for arrays that have a fixed shape.

    This is used to facilitate automatic input checking for all the different
    internal parameters. Each instance of this class can be removed without
    replacement and the code should still work, provided the user provides
    only correct angles to below parameter classes

    Parameters
    ----------
    shape: `Callable[[Any], Tuple]`
        Returns the shape for self.values

    Example
    -------
    With this descriptor, the two following are equivalent:

    .. code-block:: python

        class foo():
            def __init__(self):
                self.shape = (n, m)
                self._my_attribute = None

            @property
            def my_attribute(self):
                return _my_attribute

            @my_attribute.setter
            def my_attribute(self):
                try:
                    self._my_attribute = np.reshape(values, self.shape)
                except ValueError:
                    raise ValueError("my_attribute must have shape "
                                    f"{self.shape}")


    can be simplified to

    .. code-block:: python

        class foo():
            def __init__(self):
                self.shape = (n, m)

            @shapedArray
            def my_attribute(self):
                return self.shape
    """

    def __init__(self, shape: Callable[[Any], Tuple]):
        """The constructor. See class documentation"""
        self.name = shape.__name__
        self.shape = shape

    def __set__(self, obj, values):
        """The setter with input checking."""
        try:
            # also round to 12 decimal places to avoid floating point errors
            setattr(
                obj, f"__{self.name}", np.round(np.reshape(values, self.shape(obj)), 12)
            )
        except ValueError:
            raise ValueError(f"{self.name} must have shape {self.shape(obj)}")

    def __get__(self, obj, objtype):
        """The getter."""
        return getattr(obj, f"__{self.name}")


# GATEMAPLABEL #####################################


class GateMapType(Enum):
    MIXER = "MIXER"
    COST = "COST"
    FIXED = "FIXED"

    @classmethod
    def supported_types(cls):
        return list(map(lambda c: c.name, cls))


class GateMapLabel:
    """
    This object helps keeps track of labels associated with
    gates for their identification in the circuit.
    """

    def __init__(
        self,
        n_qubits: int = None,
        layer_number: int = None,
        application_sequence: int = None,
        gatemap_type: GateMapType = None,
    ):
        """
        Parameters
        ----------
        layer_number: `int`
            The label for the algorthmic layer
        application_sequence: `int`
            The label for the sequence of application
            for the gate
        gate_type: `GateMapType`
            Gate type for distinguishing gate between different QAOA blocks,
            and between parameterized or non-parameterized
        n_qubits: `int`
            Number of qubits in the gate
        """
        if (
            isinstance(layer_number, (int, type(None)))
            and isinstance(application_sequence, (int, type(None)))
            and isinstance(gatemap_type, (GateMapType, type(None)))
        ):
            self.layer = layer_number
            self.sequence = application_sequence
            self.type = gatemap_type
            self.n_qubits = n_qubits
        else:
            raise ValueError("Some or all of input types are incorrect")

    def __repr__(self):
        """
        String representation of the Gatemap label
        """
        representation = f"{self.n_qubits}Q_" if self.n_qubits is not None else ""
        representation += f"{self.type.value}" if self.type.value is not None else ""
        representation += f"_seq{self.sequence}" if self.sequence is not None else ""
        representation += f"_layer{self.layer}" if self.layer is not None else ""

        return representation

    def update_gatelabel(
        self,
        new_layer_number: int = None,
        new_application_sequence: int = None,
        new_gatemap_type: GateMapType = None,
    ) -> None:
        """
        Change the properties of the gatemap label to update
        the gate identity
        """
        if (
            new_layer_number is None
            and new_gatemap_type is None
            and new_application_sequence is None
        ):
            raise ValueError(
                "Pass atleast one updated attribute to update the gatemap label"
            )
        else:
            if isinstance(new_layer_number, int):
                self.layer = new_layer_number
            if isinstance(new_application_sequence, int):
                self.sequence = new_application_sequence
            if isinstance(new_gatemap_type, GateMapType):
                self.type = new_gatemap_type


# ROTATIONANGLE #####################################


class RotationAngle(object):
    def __init__(
        self,
        angle_relationship: Callable,
        gate_label: GateMapLabel,
        value: Union[int, float] = None,
    ):
        """
        Angle object as placeholder for assigning angles to the parameterized
        gates in the circuit

        Parameters
        ----------
        angle_relationship: `Callable`
            A function that takes input a parameter and assigns
            the angle to the gate depending on the relationship
            between the parameter and the angle
        gate_label: `GateMapLabel`
            The label for the gatemap object to which the rotationangle
            is assigned
        value: `int` or `float`
            Value of the parameter
        """

        self._angle = angle_relationship
        self.gate_label = gate_label
        self.value = value

    @property
    def rotation_angle(self) -> Union[int, float]:
        return self._angle(self.value)


# GATEMAP ##########################################


class GateMap(ABC):
    def __init__(self, qubit_1: int):
        self.qubit_1 = qubit_1
        self.gate_label = None

    def decomposition(self, decomposition_type: str) -> List[Tuple]:
        try:
            return getattr(self, "_decomposition_" + decomposition_type)
        except Exception as e:
            print(e, "\nReturning default decomposition.")
            return getattr(self, "_decomposition_standard")

    @property
    @abstractmethod
    def _decomposition_standard(self) -> List[Tuple]:
        pass


class SWAPGateMap(GateMap):
    def __init__(self, qubit_1: int, qubit_2: int):
        super().__init__(qubit_1)
        self.qubit_2 = qubit_2
        self.gate_label = GateMapLabel(n_qubits=2, gatemap_type=GateMapType.FIXED)

    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (CX, [self.qubit_1, self.qubit_2]),
            (CX, [self.qubit_2, self.qubit_1]),
            (CX, [self.qubit_1, self.qubit_2]),
        ]

    @property
    def _decomposition_standard2(self) -> List[Tuple]:
        return [
            (
                RZ,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (
                RZ,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            # X gate decomposition
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            # X gate decomposition
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            (
                RiSWAP,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, np.pi),
                ],
            ),
            (CZ, [self.qubit_1, self.qubit_2]),
            # X gate decomposition
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            # X gate decomposition
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
        ]


class RotationGateMap(GateMap):
    def __init__(self, qubit_1: int):
        super().__init__(qubit_1)
        self.angle_value = None
        self.gate_label = GateMapLabel(n_qubits=1)

    @property
    def _decomposition_trivial(self) -> List[Tuple]:
        return self._decomposition_standard


class RYGateMap(RotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (
                RY,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RXGateMap(RotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (
                RX,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RZGateMap(RotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (
                RZ,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class TwoQubitRotationGateMap(RotationGateMap):
    def __init__(self, qubit_1: int, qubit_2: int):
        super().__init__(qubit_1)
        self.qubit_2 = qubit_2
        self.gate_label = GateMapLabel(n_qubits=2)

    @property
    def _decomposition_trivial(self) -> List[Tuple]:
        low_level_gate = eval(type(self).__name__.strip("GateMap"))
        return [
            (
                low_level_gate,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RXXGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (
                RY,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RY,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
        ]


class RXYGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        raise NotImplementedError()


class RYYGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (
                RX,
                [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
            (
                RX,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)],
            ),
        ]


class RZXGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RY,
                [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)],
            ),
            (RX, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi)]),
        ]


class RZZGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return [
            (CX, [self.qubit_1, self.qubit_2]),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (CX, [self.qubit_1, self.qubit_2]),
        ]

    @property
    def _decomposition_standard2(self) -> List[Tuple]:
        return [
            (
                RZ,
                [
                    self.qubit_1,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (
                RZ,
                [
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            ),
            (
                CPHASE,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: -2 * x, self.gate_label, self.angle_value),
                ],
            ),
        ]


class RYZGateMap(TwoQubitRotationGateMap):
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        raise NotImplementedError()


class RiSWAPGateMap(TwoQubitRotationGateMap):
    """
    Parameterised-iSWAP gate
    """

    @property
    def _decomposition_standard(self) -> List[Tuple]:
        total_decomp = RXXGateMap.decomposition
        total_decomp.extend(RYYGateMap.decomposition)
        return total_decomp

    @property
    def _decomposition_standard2(self) -> List[Tuple]:
        return [
            (
                RiSWAP,
                [
                    self.qubit_1,
                    self.qubit_2,
                    RotationAngle(lambda x: x, self.gate_label, self.angle_value),
                ],
            )
        ]


class RotationGateMapFactory(object):
    PAULI_OPERATORS = ["X", "Y", "Z", "XX", "ZX", "ZZ", "XY", "YY", "YZ"]
    GATE_GENERATOR_GATEMAP_MAPPER = {
        term: eval(f"R{term}GateMap") for term in PAULI_OPERATORS
    }

    def rotationgatemap_list_from_hamiltonian(
        hamil_obj: Hamiltonian, gatemap_type: GateMapType = None
    ) -> List[RotationGateMap]:
        """
        Constructs a list of Rotation GateMaps from the input Hamiltonian Object.

        Parameters
        ----------
        hamil_obj: Hamiltonian
            Hamiltonian object to construct the circuit from
        gatemap_type: GateMapType
            Gatemap type constructed
        """

        pauli_terms = hamil_obj.terms
        output_gates = []

        one_qubit_count = 0
        two_qubit_count = 0

        for each_term in pauli_terms:
            if each_term.pauli_str in RotationGateMapFactory.PAULI_OPERATORS:
                pauli_str = each_term.pauli_str
                qubit_indices = each_term.qubit_indices
            elif each_term.pauli_str[::-1] in RotationGateMapFactory.PAULI_OPERATORS:
                pauli_str = each_term.pauli_str[::-1]
                qubit_indices = each_term.qubit_indices[::-1]
            else:
                raise ValueError("Hamiltonian contains non-Pauli terms")

            try:
                gate_class = RotationGateMapFactory.GATE_GENERATOR_GATEMAP_MAPPER[
                    pauli_str
                ]
            except Exception:
                raise Exception("Generating gates from Hamiltonian terms failed")

            if len(each_term.qubit_indices) == 2:
                gate = gate_class(qubit_indices[0], qubit_indices[1])
                gate.gate_label.update_gatelabel(
                    new_application_sequence=two_qubit_count,
                    new_gatemap_type=gatemap_type,
                )
                output_gates.append(gate)
                two_qubit_count += 1
            elif len(each_term.qubit_indices) == 1:
                gate = gate_class(qubit_indices[0])
                gate.gate_label.update_gatelabel(
                    new_application_sequence=one_qubit_count,
                    new_gatemap_type=gatemap_type,
                )
                output_gates.append(gate)
                one_qubit_count += 1

        return output_gates

    def gatemaps_layer_relabel(
        gatemap_list: List[GateMap], new_layer_number: int
    ) -> List[RotationGateMap]:
        """
        Reconstruct a new gatemap list from a list of RotationGateMap Objects with the input
        layer number in the gate_label attribute.

        Parameters
        ----------
        gatemap_list: `List[RotationGateMap]
            The list of GateMap objects whose labels need to be udpated
        """
        output_gate_list = []

        for each_gatemap in gatemap_list:
            new_gatemap = deepcopy(each_gatemap)
            new_gatemap.gate_label.update_gatelabel(new_layer_number=new_layer_number)
            output_gate_list.append(new_gatemap)

        return output_gate_list


# GATES ############################################


class GateApplicator(ABC):
    @abstractmethod
    def apply_gate(self, gate) -> Callable:
        """
        Apply gate to the circuit
        """
        pass


class Gate(ABC):
    def __init__(self, applicator: GateApplicator):
        self.applicator = applicator

    @abstractmethod
    def apply_gate(self, ckt):
        pass


class OneQubitGate(Gate):
    def __init__(self, applicator: GateApplicator, qubit_1: int):
        super().__init__(applicator)
        self.qubit_1 = qubit_1
        self.n_qubits = 1

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, ckt)


class OneQubitRotationGate(OneQubitGate):
    def __init__(
        self, applicator: GateApplicator, qubit_1: int, rotation_object: RotationAngle
    ):
        super().__init__(applicator, qubit_1)
        self.rotation_object = rotation_object

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, self.rotation_object, ckt)


class TwoQubitGate(Gate):
    def __init__(self, applicator: GateApplicator, qubit_1: int, qubit_2: int):
        super().__init__(applicator)
        self.qubit_1 = qubit_1
        self.qubit_2 = qubit_2
        self.n_qubits = 2

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(self, self.qubit_1, self.qubit_2, ckt)


class TwoQubitRotationGate(TwoQubitGate):
    def __init__(
        self,
        applicator: GateApplicator,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
    ):
        super().__init__(applicator, qubit_1, qubit_2)
        self.rotation_object = rotation_object

    def apply_gate(self, ckt):
        return self.applicator.apply_gate(
            self, self.qubit_1, self.qubit_2, self.rotation_object, ckt
        )

    def apply_vector_gate(self, input_obj):
        input_obj.apply_rzz(
            self.qubit_1, self.qubit_2, self.rotation_object.rotation_angle
        )


class X(OneQubitGate):
    __name__ = "X"


class RZ(OneQubitRotationGate):
    __name__ = "RZ"


class RY(OneQubitRotationGate):
    __name__ = "RY"


class RX(OneQubitRotationGate):
    __name__ = "RX"


class CZ(TwoQubitGate):
    __name__ = "CZ"


class CX(TwoQubitGate):
    __name__ = "CX"


class RXX(TwoQubitRotationGate):
    __name__ = "RXX"


class RYY(TwoQubitRotationGate):
    __name__ = "RYY"


class RZZ(TwoQubitRotationGate):
    __name__ = "RZZ"


class RXY(TwoQubitRotationGate):
    __name__ = "RXY"


class RZX(TwoQubitRotationGate):
    __name__ = "RZX"


class RYZ(TwoQubitRotationGate):
    __name__ = "RYZ"


class CPHASE(TwoQubitRotationGate):
    __name__ = "CPHASE"


class RiSWAP(TwoQubitRotationGate):
    __name__ = "RiSWAP"


# HAMILTONIANMAPPER #################################


class HamiltonianMapper(object):
    def generate_gate_maps(
        hamil_obj: Hamiltonian, gatemap_type: GateMapType
    ) -> List[GateMap]:
        """
        This method gets the rotation gates based on the input Hamiltonian into the Mapper

        Parameters
        ----------
        hamil_obj : `Hamiltonian`
            The Hamiltonian object to construct the gates from
        input_label : `GateMapType`
            Input label defining the type of gate

        Returns
        -------
        `list[GateMap]`
            List of RotationGateMap objects defining part of the circuit
        """
        assert isinstance(
            gatemap_type, GateMapType
        ), f"gatemap_type must be of supported types: {GateMapType.supported_types}"
        return RotationGateMapFactory.rotationgatemap_list_from_hamiltonian(
            hamil_obj, gatemap_type
        )

    def repeat_gate_maps(
        gatemap_list: List[GateMap], n_layers: int
    ) -> List[List[GateMap]]:
        """
        Repeat the gates for n_layers based on the input gatelist

        Parameters
        ----------
        gatemap_list : `List[GateMap]`
            Repeat the gates from the gatemap_list
        n_layers: `int`
            The number of times the layer of gates have to be repeated.
        """
        output_gate_list = []
        for each_layer in range(n_layers):
            output_gate_list.append(
                RotationGateMapFactory.gatemaps_layer_relabel(gatemap_list, each_layer)
            )

        return output_gate_list


class AnsatzDescriptor(ABC):
    """
    Parameters class to construct a specific quantum ansatz to attack
    a problem

    Parameters
    ----------
    algorithm: `str`
        The algorithm corresponding to the ansatz
    Attributes
    ----------
    algorithm: `str`
    """

    def __init__(self, algorithm: str):
        self.algorithm = algorithm

    @abstractproperty
    def n_qubits(self) -> int:
        pass


class QAOADescriptor(AnsatzDescriptor):

    """
    Create the problem attributes consisting of the Hamiltonian, QAOA 'p'
    value and other specific parameters.

    Attributes
    ----------
    cost_hamiltonian: `Hamiltonian`

    qureg: `List[int]`

    cost_block_coeffs: `List[float]`

    cost_single_qubit_coeffs: `List[float]`

    cost_qubits_singles: `List[str]`

    cost_pair_qubit_coeffs: `List[float]`

    cost_qubits_pairs: `List[str]`

    mixer_block_coeffs: `List[float]`

    cost_blocks: `List[RotationGateMap]`

    mixer_blocks: `List[RotationGateMap]`

    Properties
    ----------
    n_qubits: `int`

    abstract_circuit: `List[RotationGateMap]`
    """

    def __init__(
        self,
        cost_hamiltonian: Hamiltonian,
        mixer_block: Union[List, Hamiltonian],
        p: int,
        mixer_coeffs: List[float] = [],
        routing_function: Optional[Callable] = None,
        device=None,
    ):
        """
        Parameters
        ----------
        cost_hamiltonian: `Hamiltonian`
            The cost hamiltonian of the problem the user is trying to solve.
        mixer_block: Union[List[RotationGateMap], Hamiltonian]
            The mixer hamiltonian or a list of initialised RotationGateMap objects
            that defines the gates to be used within the "mixer part" of the circuit.
        p: `int`
            Number of QAOA layers; defaults to 1 if not specified
        mixer_coeffs: `List[float]`
            A list containing coefficients for each mixer GateMap. The order of the
            coefficients should follow the order of the GateMaps provided in the relevant gate block.
            This input isnt required if the input mixer block is of type Hamiltonian.
        routing_function Optional[Callable]
            A callable function running the routing algorithm on the problem
        device
            The device on which to run the Quantum Circuit
        """

        super().__init__(algorithm="QAOA")

        self.p = p
        self.cost_block_coeffs = cost_hamiltonian.coeffs

        try:
            self.mixer_block_coeffs = mixer_block.coeffs
        except AttributeError:
            self.mixer_block_coeffs = mixer_coeffs

        # Needed in the BaseBackend to compute exact_solution, cost_funtion method
        # and bitstring_energy
        self.cost_hamiltonian = cost_hamiltonian
        self.cost_block = self.block_setter(cost_hamiltonian, GateMapType.COST)
        (
            self.cost_single_qubit_coeffs,
            self.cost_pair_qubit_coeffs,
            self.cost_qubits_singles,
            self.cost_qubits_pairs,
        ) = self._assign_coefficients(self.cost_block, self.cost_block_coeffs)

        # route the cost block and append SWAP gates
        if isinstance(routing_function, Callable):
            try:
                (
                    self.cost_block,
                    self.initial_mapping,
                    self.final_mapping,
                ) = self.route_gates_list(self.cost_block, device, routing_function)
                self.routed = True
            except TypeError:
                raise TypeError(
                    "The specified function can has a set signature that accepts"
                    " device, problem, and initial_mapping"
                )
            except Exception as e:
                raise e
        elif routing_function == None:
            self.routed = False
        else:
            raise ValueError(
                f"Routing function can only be a Callable not {type(routing_function)}"
            )

        self.mixer_block = self.block_setter(mixer_block, GateMapType.MIXER)
        (
            self.mixer_single_qubit_coeffs,
            self.mixer_pair_qubit_coeffs,
            self.mixer_qubits_singles,
            self.mixer_qubits_pairs,
        ) = self._assign_coefficients(self.mixer_block, self.mixer_block_coeffs)

        self.mixer_blocks = HamiltonianMapper.repeat_gate_maps(self.mixer_block, self.p)
        self.cost_blocks = HamiltonianMapper.repeat_gate_maps(self.cost_block, self.p)
        self.qureg = list(range(self.n_qubits))

    @property
    def n_qubits(self) -> int:
        if self.routed == True:
            return len(self.final_mapping)
        else:
            return self.cost_hamiltonian.n_qubits

    def __repr__(self):
        """Return an overview over the parameters and hyperparameters

        Todo
        ----
        Split this into ``__repr__`` and ``__str__`` with a more verbose
        output in ``__repr__``.
        """

        string = "Circuit Parameters:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "\tregister: " + str(self.qureg) + "\n" + "\n"

        string += "Cost Hamiltonian:\n"
        string += "\tcost_qubits_singles: " + str(self.cost_qubits_singles) + "\n"
        string += (
            "\tcost_single_qubit_coeffs: " + str(self.cost_single_qubit_coeffs) + "\n"
        )
        string += "\tcost_qubits_pairs: " + str(self.cost_qubits_pairs) + "\n"
        string += (
            "\tcost_pair_qubit_coeffs: "
            + str(self.cost_pair_qubit_coeffs)
            + "\n"
            + "\n"
        )

        string += "Mixer Hamiltonian:\n"
        string += "\tmixer_qubits_singles: " + str(self.mixer_qubits_singles) + "\n"
        string += (
            "\tmixer_single_qubit_coeffs: " + str(self.mixer_single_qubit_coeffs) + "\n"
        )
        string += "\tmixer_qubits_pairs: " + str(self.mixer_qubits_pairs) + "\n"
        string += (
            "\tmixer_pair_qubit_coeffs: " + str(self.mixer_pair_qubit_coeffs) + "\n"
        )

        return string

    def _assign_coefficients(
        self, input_block: List[RotationGateMap], input_coeffs: List[float]
    ) -> None:
        """
        Splits the coefficients and gatemaps into qubit singles and qubit pairs.
        """

        single_qubit_coeffs = []
        pair_qubit_coeffs = []
        qubit_singles = []
        qubit_pairs = []

        if len(input_block) != len(input_coeffs):
            raise ValueError(
                "The number of terms/gatemaps must match the number of coefficients provided."
            )
        for each_gatemap, each_coeff in zip(input_block, input_coeffs):
            if each_gatemap.gate_label.n_qubits == 1:
                single_qubit_coeffs.append(each_coeff)
                # Giving a string name to each gatemap (?)
                qubit_singles.append(type(each_gatemap).__name__)
            elif each_gatemap.gate_label.n_qubits == 2:
                pair_qubit_coeffs.append(each_coeff)
                qubit_pairs.append(type(each_gatemap).__name__)

        return (single_qubit_coeffs, pair_qubit_coeffs, qubit_singles, qubit_pairs)

    @staticmethod
    def block_setter(
        input_object: Union[List["RotationGateMap"], Hamiltonian], block_type: Enum
    ) -> List["RotationGateMap"]:
        """
        Converts a Hamiltonian Object into a List of RotationGateMap Objects with
        the appropriate block_type and sequence assigned to the GateLabel

        OR

        Remaps a list of RotationGateMap Objects with a block_type and sequence
        implied from its position in the list.

        Parameters
        ----------
        input_object: `Union[List[RotationGateMap], Hamiltonian]`
            A Hamiltonian Object or a list of RotationGateMap Objects (Ordered
            according to their application order in the final circuit)
        block_type: Enum
            The type to be assigned to all the RotationGateMap Objects generated
            from input_object

        Returns
        -------
        `List[RotationGateMap]`
        """

        if isinstance(input_object, Hamiltonian):
            block = HamiltonianMapper.generate_gate_maps(input_object, block_type)
        elif isinstance(input_object, list):
            input_object = QAOADescriptor.set_block_sequence(input_object)
            for each_gate in input_object:
                if isinstance(each_gate, RotationGateMap):
                    each_gate.gate_label.update_gatelabel(new_gatemap_type=block_type)
                else:
                    raise TypeError(
                        f"Input gate is of unsupported type {type(each_gate)}."
                        "Only RotationGateMaps are supported"
                    )
            block = input_object
        else:
            raise ValueError(
                "The input object defining mixer should be a List of RotationGateMaps or type Hamiltonian"
            )
        return block

    @staticmethod
    def set_block_sequence(
        input_gatemap_list: List["RotationGateMap"],
    ) -> List["RotationGateMap"]:
        """
        This method assigns the sequence attribute to all RotationGateMap objects in the list.
        The sequence of the GateMaps are implied based on their positions in the list.

        Parameters
        ----------
        input_gatemap_list: `List[RotationGateMap]`
            A list of RotationGateMap Objects

        Returns
        -------
        `List[RotationGateMap]`
        """

        one_qubit_count = 0
        two_qubit_count = 0

        for each_gate in input_gatemap_list:
            if isinstance(each_gate, RotationGateMap):
                if each_gate.gate_label.n_qubits == 1:
                    each_gate.gate_label.update_gatelabel(
                        new_application_sequence=one_qubit_count,
                    )
                    one_qubit_count += 1
                elif each_gate.gate_label.n_qubits == 2:
                    each_gate.gate_label.update_gatelabel(
                        new_application_sequence=two_qubit_count,
                    )
                    two_qubit_count += 1
            else:
                raise TypeError(
                    f"Input gate is of unsupported type {type(each_gate)}."
                    "Only RotationGateMaps are supported"
                )
        return input_gatemap_list

    def reorder_gates_block(self, gates_block, layer_number):
        """Update the qubits that the gates are acting on after application
        of SWAPs in the cost layer
        """
        for gate in gates_block:
            if layer_number % 2 == 0:
                mapping = self.final_mapping
                gate.qubit_1 = mapping[gate.qubit_1]
                if gate.gate_label.n_qubits == 2:
                    gate.qubit_2 = mapping[gate.qubit_2]
            else:
                pass

        return gates_block

    @staticmethod
    def route_gates_list(
        gates_to_route: List["GateMap"],
        device,
        routing_function: Callable,
    ) -> List["GateMap"]:
        """
        Apply qubit routing to the abstract circuit gate list
        based on device information

        Parameters
        ----------
        gates_to_route: `List[GateMap]`
            The gates to route
        device
            The device on which to run the circuit
        routing_function: `Callable`
            The function that accepts as input the device, problem, initial_mapping and
            outputs the list of gates with swaps
        """
        original_qubits_to_gate_mapping = {
            (gate.qubit_1, gate.qubit_2): gate
            for gate in gates_to_route
            if gate.gate_label.n_qubits == 2
        }
        problem_to_solve = list(original_qubits_to_gate_mapping.keys())
        (
            gate_list_indices,
            swap_mask,
            initial_physical_to_logical_mapping,
            final_mapping,
        ) = routing_function(device, problem_to_solve)

        gates_list = [gate for gate in gates_to_route if gate.gate_label.n_qubits == 1]
        swapped_history = []
        for idx, pair_ij in enumerate(gate_list_indices):
            mask = swap_mask[idx]
            qi, qj = pair_ij
            if mask == True:
                swapped_history.append(pair_ij)
                gates_list.append(SWAPGateMap(qi, qj))
            elif mask == False:
                old_qi, old_qj = qi, qj
                # traverse each SWAP application in reverse order to obtain
                # the original location of the current qubit
                for swap_pair in swapped_history[::-1]:
                    if old_qi in swap_pair:
                        old_qi = (
                            swap_pair[0] if swap_pair[1] == old_qi else swap_pair[1]
                        )
                    if old_qj in swap_pair:
                        old_qj = (
                            swap_pair[0] if swap_pair[1] == old_qj else swap_pair[1]
                        )
                try:
                    ising_gate = original_qubits_to_gate_mapping[
                        tuple([old_qi, old_qj])
                    ]
                except KeyError:
                    ising_gate = original_qubits_to_gate_mapping[
                        tuple([old_qj, old_qi])
                    ]
                except Exception as e:
                    raise e
                ising_gate.qubit_1, ising_gate.qubit_2 = qi, qj
                gates_list.append(ising_gate)

        return (
            gates_list,
            list(initial_physical_to_logical_mapping.keys()),
            final_mapping,
        )

    @property
    def abstract_circuit(self):
        # even layer inversion if the circuit contains SWAP gates
        even_layer_inversion = -1 if self.routed == True else 1
        _abstract_circuit = []
        for each_p in range(self.p):
            # apply each cost_block with reversed order to maintain the SWAP sequence
            _abstract_circuit.extend(
                self.cost_blocks[each_p][:: (even_layer_inversion) ** each_p]
            )
            # apply the mixer block
            if self.routed == True:
                mixer_block = self.reorder_gates_block(
                    self.mixer_blocks[each_p], each_p
                )
            else:
                mixer_block = self.mixer_blocks[each_p]
            _abstract_circuit.extend(mixer_block)

        return _abstract_circuit


# VARIATIONAL BASEPARAMS ############################


class QAOAVariationalBaseParams(ABC):
    """
    A class that initialises and keeps track of the Variational
    parameters

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        Specify the circuit parameters to construct circuit angles to be
        used for training

    Attributes
    ----------
    qaoa_descriptor: `QAOADescriptor`
    p: `int`
    cost_1q_coeffs
    cost_2q_coeffs
    mixer_1q_coeffs
    mixer_2q_coeffs
    """

    def __init__(self, qaoa_descriptor: QAOADescriptor):
        self.qaoa_descriptor = qaoa_descriptor
        self.p = self.qaoa_descriptor.p

        try:
            self.cost_1q_coeffs = qaoa_descriptor.cost_single_qubit_coeffs
            self.cost_2q_coeffs = qaoa_descriptor.cost_pair_qubit_coeffs
            self.mixer_1q_coeffs = qaoa_descriptor.mixer_single_qubit_coeffs
            self.mixer_2q_coeffs = qaoa_descriptor.mixer_pair_qubit_coeffs
        except AttributeError:
            self.cost_1q_coeffs = qaoa_descriptor.cost_hamiltonian.single_qubit_coeffs
            self.cost_2q_coeffs = qaoa_descriptor.cost_hamiltonian.pair_qubit_coeffs
            self.mixer_1q_coeffs = qaoa_descriptor.mixer_hamiltonian.single_qubit_coeffs
            self.mixer_2q_coeffs = qaoa_descriptor.mixer_hamiltonian.pair_qubit_coeffs

    def __len__(self):
        """
        Returns
        -------
        int:
            the length of the data produced by self.raw() and accepted by
            self.update_from_raw()
        """
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __str__(self):
        return self.__repr__()

    @property
    def mixer_1q_angles(self) -> np.ndarray:
        """2D array with the X-rotation angles.

        1st index goes over p and the 2nd index over the qubits to
        apply X-rotations on.
        """
        raise NotImplementedError()

    @property
    def mixer_2q_angles(self) -> np.ndarray:
        """2D array with the X-rotation angles.

        1st index goes over p and the 2nd index over the qubits to
        apply X-rotations on.
        """
        raise NotImplementedError()

    @property
    def cost_1q_angles(self) -> np.ndarray:
        """2D array with the ZZ-rotation angles.

        1st index goes over the p and the 2nd index over the qubit
        pairs, to apply ZZ-rotations on.
        """
        raise NotImplementedError()

    @property
    def cost_2q_angles(self) -> np.ndarray:
        """2D array with Z-rotation angles.

        1st index goes over the p and the 2nd index over the qubit
        pairs, to apply Z-rotations on. These are needed by
        ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError()

    def update_from_raw(self, new_values: Union[list, np.array]):
        """
        Update all the parameters from a 1D array.

        The input has the same format as the output of ``self.raw()``.
        This is useful for ``scipy.optimize.minimize`` which expects
        the parameters that need to be optimized to be a 1D array.

        Parameters
        ----------
        new_values: `Union[list, np.array]`
            A 1D array with the new parameters. Must have length  ``len(self)``
            and the ordering of the flattend ``parameters`` in ``__init__()``.

        """
        raise NotImplementedError()

    def raw(self) -> np.ndarray:
        """
        Return the parameters in a 1D array.

        This 1D array is needed by ``scipy.optimize.minimize`` which expects
        the parameters that need to be optimized to be a 1D array.

        Returns
        -------
        np.array:
            The parameters in a 1D array. Has the same output format as the
            expected input of ``self.update_from_raw``. Hence corresponds to
            the flattened `parameters` in `__init__()`

        """
        raise NotImplementedError()

    def update_from_dict(self, new_values: dict):
        """
        Update all the parameters from a dictionary.

        The input has the same format as the output of ``self.asdict()``.

        Parameters
        ----------
        new_values: `dict`
            A dictionary with the new parameters. Must have the same keys as
            the output of ``self.asdict()``.

        """

        assert isinstance(new_values, dict), f"Expected dict, got {type(new_values)}"

        for key, value in new_values.items():
            if key not in self.asdict().keys():
                raise KeyError(
                    f"'{key}' not in {self.__class__.__name__}, expected keys: {list(self.asdict().keys())}"
                )
            else:
                if getattr(self, key).shape != np.array(value).shape:
                    raise ValueError(
                        f"Shape of '{key}' does not match. Expected shape {getattr(self, key).shape}, got {np.array(value).shape}."
                    )

        raw_params = []
        for key, value in self.asdict().items():
            if key in new_values.keys():
                raw_params += list(np.array(new_values[key]).flatten())
            else:
                raw_params += list(np.array(value).flatten())

        self.update_from_raw(raw_params)

    def asdict(self) -> dict:
        """
        Return the parameters as a dictionary.

        Returns
        -------
        dict:
            The parameters as a dictionary. Has the same output format as the
            expected input of ``self.update_from_dict``.

        """
        return {k[2:]: v for k, v in self.__dict__.items() if k[0:2] == "__"}

    @classmethod
    def linear_ramp_from_hamiltonian(
        cls, qaoa_descriptor: QAOADescriptor, time: float = None
    ):
        """Alternative to ``__init__`` that already fills ``parameters``.

        Calculate initial parameters from register, terms, weights
        (specifiying a Hamiltonian), corresponding to a linear ramp
        annealing schedule and return a ``QAOAVariationalBaseParams`` object.

        Parameters
        ----------
        qaoa_descriptor: `QAOADescriptor`
            QAOADescriptor object containing information about terms,weights,register and p

        time: `float`
            Total annealing time. Defaults to ``0.7*p``.

        Returns
        -------
        QAOAVariationalBaseParams:
            The initial parameters for a linear ramp for ``hamiltonian``.

        """
        raise NotImplementedError()

    @classmethod
    def random(cls, qaoa_descriptor: QAOADescriptor, seed: int = None):
        """
        Initialise parameters randomly

        Parameters
        ----------
        qaoa_descriptor: `QAOADescriptor`
            QAOADescriptor object containing information about terms,
            weights, register and p.

        seed: `int`
                Use a fixed seed for reproducible random numbers

        Returns
        -------
        QAOAVariationalBaseParams:
            Randomly initialiased parameters
        """
        raise NotImplementedError()

    @classmethod
    def empty(cls, qaoa_descriptor: QAOADescriptor):
        """
        Alternative to ``__init__`` that only takes ``qaoa_descriptor`` and
        fills ``parameters`` via ``np.empty``

        Parameters
        ----------
        qaoa_descriptor: `QAOADescriptor`
            QAOADescriptor object containing information about terms,weights,register and p

        Returns
        -------
        QAOAVariationalBaseParams:
            A Parameter object with the parameters filled by ``np.empty``
        """
        raise NotImplementedError()

    @classmethod
    def from_other_parameters(cls, params):
        """Alternative to ``__init__`` that takes parameters with less degrees
        of freedom as the input.

        Parameters
        ----------
        params: `QAOAVaritionalBaseParams`
            The input parameters object to construct the new parameters object from.
        Returns
        -------
        QAOAVariationalBaseParams:
            The converted paramters s.t. all the rotation angles of the in
            and output parameters are the same.
        """
        from . import converter

        return converter(params, cls)

    def raw_rotation_angles(self) -> np.ndarray:
        """
        Flat array of the rotation angles for the memory map for the
        parametric circuit.

        Returns
        -------
        np.array:
            Returns all single rotation angles in the ordering
            ``(x_rotation_angles, gamma_singles, zz_rotation_angles)`` where
            ``x_rotation_angles = (beta_q0_t0, beta_q1_t0, ... , beta_qn_tp)``
            and the same for ``z_rotation_angles`` and ``zz_rotation_angles``

        """
        raw_data = np.concatenate(
            (
                self.mixer_1q_angles.flatten(),
                self.mixer_2q_angles.flatten(),
                self.cost_1q_angles.flatten(),
                self.cost_1q_angles.flatten(),
            )
        )
        return raw_data

    def plot(self, ax=None, **kwargs):
        """
        Plots ``self`` in a sensible way to the canvas ``ax``, if provided.

        Parameters
        ----------
        ax: `matplotlib.axes._subplots.AxesSubplot`
                The canvas to plot itself on
        kwargs:
                All remaining keyword arguments are passed forward to the plot
                function

        """
        raise NotImplementedError()


class QAOAParameterIterator:
    """An iterator to sweep one parameter over a range in a QAOAParameter object.

    Parameters
    ----------
    qaoa_params:
        The initial QAOA parameters, where one of them is swept over
    the_parameter:
        A string specifying, which parameter should be varied. It has to be
        of the form ``<attr_name>[i]`` where ``<attr_name>`` is the name
        of the _internal_ list and ``i`` the index, at which it sits. E.g.
        if ``qaoa_params`` is of type ``AnnealingParams``
        and  we want to vary over the second timestep, it is
        ``the_parameter = "times[1]"``.
    the_range:
        The range, that ``the_parameter`` should be varied over

    Todo
    ----
    - Add checks, that the number of indices in ``the_parameter`` matches
      the dimensions of ``the_parameter``
    - Add checks, that the index is not too large

    Example
    -------
    Assume qaoa_params is of type ``StandardWithBiasParams`` and
    has `p >= 2`. Then the following code produces a loop that
    sweeps ``gammas_singles[1]`` over the range ``(0, 1)`` in 4 layers:

    .. code-block:: python

        the_range = np.arange(0, 1, 0.4)
        the_parameter = "gammas_singles[1]"
        param_iterator = QAOAParameterIterator(qaoa_params, the_parameter, the_range)
        for params in param_iterator:
            # do what ever needs to be done.
            # we have type(params) == type(qaoa_params)
    """

    def __init__(
        self,
        variational_params: QAOAVariationalBaseParams,
        the_parameter: str,
        the_range: Iterable[float],
    ):
        """See class documentation for details"""
        self.params = variational_params
        self.iterator = iter(the_range)
        self.the_parameter, *indices = the_parameter.split("[")
        indices = [i.replace("]", "") for i in indices]
        if len(indices) == 1:
            self.index0 = int(indices[0])
            self.index1 = False
        elif len(indices) == 2:
            self.index0 = int(indices[0])
            self.index1 = int(indices[1])
        else:
            raise ValueError("the_parameter has to many indices")

    def __iter__(self):
        return self

    def __next__(self):
        # get next value from the_range
        value = next(self.iterator)

        # 2d list or 1d list?
        if self.index1 is not False:
            getattr(self.params, self.the_parameter)[self.index0][self.index1] = value
        else:
            getattr(self.params, self.the_parameter)[self.index0] = value

        return self.params


# STANDARD PARAMS ###################################


class QAOAVariationalStandardParams(QAOAVariationalBaseParams):
    r"""
    QAOA parameters that implement a state preparation circuit with

    .. math::

        e^{-i \beta_p H_0}
        e^{-i \gamma_p H_c}
        \cdots
        e^{-i \beta_0 H_0}
        e^{-i \gamma_0 H_c}

    This corresponds to the parametrization used by Farhi in his
    original paper [https://arxiv.org/abs/1411.4028]

    Parameters
    ----------
    qaoa_descriptor:
        QAOADescriptor object containing circuit instructions
    betas:
        List of p betas
    gammas:
        List of p gammas

    Attributes
    ----------
    betas: np.array
        1D array with the betas from above
    gammas: np.array
        1D array with the gamma from above
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        betas: List[Union[float, int]],
        gammas: List[Union[float, int]],
    ):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_descriptor)
        self.betas = np.array(betas)
        self.gammas = np.array(gammas)

    def __repr__(self):
        string = "Standard Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Variational Parameters:\n"
        string += "\tbetas: " + str(self.betas) + "\n"
        string += "\tgammas: " + str(self.gammas) + "\n"
        return string

    def __len__(self):
        return self.p * 2

    @shapedArray
    def betas(self):
        return self.p

    @shapedArray
    def gammas(self):
        return self.p

    @property
    def mixer_1q_angles(self):
        return 2 * np.outer(self.betas, self.mixer_1q_coeffs)

    @property
    def mixer_2q_angles(self):
        return 2 * np.outer(self.betas, self.mixer_2q_coeffs)

    @property
    def cost_1q_angles(self):
        return 2 * np.outer(self.gammas, self.cost_1q_coeffs)

    @property
    def cost_2q_angles(self):
        return 2 * np.outer(self.gammas, self.cost_2q_coeffs)

    def update_from_raw(self, new_values):
        # overwrite self.betas with new ones
        self.betas = np.array(new_values[0 : self.p])
        new_values = new_values[self.p :]  # cut betas from new_values
        self.gammas = np.array(new_values[0 : self.p])
        new_values = new_values[self.p :]

        if len(new_values) != 0:
            raise RuntimeWarning(
                "Incorrect dimension specified for new_values"
                "to construct the new betas and new gammas"
            )

    def raw(self):
        raw_data = np.concatenate((self.betas, self.gammas))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(
        cls, qaoa_descriptor: QAOADescriptor, time: float = None
    ):
        """
        Returns
        -------
        StandardParams
            A ``StandardParams`` object with parameters according
            to a linear ramp schedule for the Hamiltonian specified by register, terms, weights.
        """
        p = qaoa_descriptor.p

        if time is None:
            time = float(0.7 * p)
        # create evenly spaced timelayers at the centers of p intervals
        dt = time / p
        # fill betas, gammas_singles and gammas_pairs
        betas = np.linspace(
            (dt / time) * (time * (1 - 0.5 / p)), (dt / time) * (time * 0.5 / p), p
        )
        gammas = betas[::-1]
        # wrap it all nicely in a qaoa_parameters object
        params = cls(qaoa_descriptor, betas, gammas)

        return params

    @classmethod
    def random(cls, qaoa_descriptor: QAOADescriptor, seed: int = None):
        """
        Returns
        -------
        StandardParams
            Randomly initialised ``StandardParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        betas = np.random.uniform(0, np.pi, qaoa_descriptor.p)
        gammas = np.random.uniform(0, np.pi, qaoa_descriptor.p)

        params = cls(qaoa_descriptor, betas, gammas)

        return params

    @classmethod
    def empty(cls, qaoa_descriptor: QAOADescriptor):
        """
        Initialise Standard Variational params with empty arrays
        """
        p = qaoa_descriptor.p
        betas = np.empty(p)
        gammas = np.empty(p)

        return cls(qaoa_descriptor, betas, gammas)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        ax.plot(self.gammas, label="gammas", marker="^", ls="", **kwargs)
        ax.set_xlabel("p", fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

    def convert_to_ext(self, args_std):
        """
        Method that converts a list of parameters in the standard parametrisation
        form (args_std) to an equivalent list of parameters in the extended parametrisation form.

        PARAMETERS
        ----------
        args_std :
            Parameters (a list of float) in the standard parametrisation form.

        RETURNS
        -------
        args_ext:
            Parameters (a list of float) in the extended parametrisation form.

        """

        terms_lst = [
            len(self.mixer_1q_coeffs),
            len(self.mixer_2q_coeffs),
            len(self.cost_1q_coeffs),
            len(self.cost_2q_coeffs),
        ]
        terms_lst_p = np.repeat(terms_lst, [self.p] * len(terms_lst))
        args_ext = []
        for i in range(4):  # 4 types of terms
            for j in range(self.p):
                for k in range(terms_lst_p[i * self.p + j]):
                    if i < 2:
                        args_ext.append(args_std[j])
                    else:
                        args_ext.append(args_std[j + int(len(args_std) / 2)])

        return args_ext


class QAOAVariationalStandardWithBiasParams(QAOAVariationalBaseParams):
    r"""
    QAOA parameters that implement a state preparation circuit with

    .. math::

        e^{-i \beta_p H_0}
        e^{-i \gamma_{\textrm{singles}, p} H_{c, \textrm{singles}}}
        e^{-i \gamma_{\textrm{pairs}, p} H_{c, \textrm{pairs}}}
        \cdots
        e^{-i \beta_0 H_0}
        e^{-i \gamma_{\textrm{singles}, 0} H_{c, \textrm{singles}}}
        e^{-i \gamma_{\textrm{pairs}, 0} H_{c, \textrm{pairs}}}

    where the cost hamiltonian is split into :math:`H_{c, \textrm{singles}}`
    the bias terms, that act on only one qubit, and
    :math:`H_{c, \textrm{pairs}}` the coupling terms, that act on two qubits.

    Parameters
    ----------
    qaoa_descriptor:
        QAOADescriptor object containing circuit instructions
    betas:
        List of p betas
    gammas_singles:
        List of p gammas_singles
    gammas_pairs:
        List of p gammas_pairs

    Attributes
    ----------
    betas: np.array
        A 1D array containing the betas from above for each timestep
    gammas_pairs: np.array
        A 1D array containing the gammas_singles from above for each timestep
    gammas_singles: np.array
        A 1D array containing the gammas_pairs from above for each timestep
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        betas: List[Union[float, int]],
        gammas_singles: List[Union[float, int]],
        gammas_pairs: List[Union[float, int]],
    ):
        super().__init__(qaoa_descriptor)
        if not self.cost_1q_coeffs or not self.cost_2q_coeffs:
            raise RuntimeError(
                f"Please choose {type(self).__name__} parameterisation for "
                "problems containing both Cost One-Qubit and Two-Qubit terms"
            )

        self.betas = np.array(betas)
        self.gammas_singles = np.array(gammas_singles)
        self.gammas_pairs = np.array(gammas_pairs)

    def __repr__(self):
        string = "Standard with Bias Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Variational Parameters:\n"
        string += "\tbetas: " + str(self.betas) + "\n"
        string += "\tgammas_singles: " + str(self.gammas_singles) + "\n"
        string += "\tgammas_pairs: " + str(self.gammas_pairs) + "\n"
        return string

    def __len__(self):
        return self.p * 3

    @shapedArray
    def betas(self):
        return self.p

    @shapedArray
    def gammas_singles(self):
        return self.p

    @shapedArray
    def gammas_pairs(self):
        return self.p

    @property
    def mixer_1q_angles(self):
        return 2 * np.outer(self.betas, self.mixer_1q_coeffs)

    @property
    def mixer_2q_angles(self):
        return 2 * np.outer(self.betas, self.mixer_2q_coeffs)

    @property
    def cost_1q_angles(self):
        return 2 * np.outer(self.gammas_singles, self.cost_1q_coeffs)

    @property
    def cost_2q_angles(self):
        return 2 * np.outer(self.gammas_pairs, self.cost_2q_coeffs)

    def update_from_raw(self, new_values):
        # overwrite self.betas with new ones
        self.betas = np.array(new_values[0 : self.p])
        new_values = new_values[self.p :]  # cut betas from new_values
        self.gammas_singles = np.array(new_values[0 : self.p])
        new_values = new_values[self.p :]
        self.gammas_pairs = np.array(new_values[0 : self.p])
        new_values = new_values[self.p :]

        if len(new_values) != 0:
            raise RuntimeWarning(
                "Incorrect dimension specified for new_values"
                "to construct the new betas and new gammas"
            )

    def raw(self):
        raw_data = np.concatenate((self.betas, self.gammas_singles, self.gammas_pairs))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(
        cls, qaoa_descriptor: QAOADescriptor, time: float = None
    ):
        """
        Returns
        -------
        StandardParams
            A ``StandardParams`` object with parameters according
            to a linear ramp schedule for the Hamiltonian specified by register, terms, weights.
        """
        p = qaoa_descriptor.p

        if time is None:
            time = float(0.7 * p)
        # create evenly spaced timelayers at the centers of p intervals
        dt = time / p
        # fill betas, gammas_singles and gammas_pairs
        betas = np.linspace(
            (dt / time) * (time * (1 - 0.5 / p)), (dt / time) * (time * 0.5 / p), p
        )
        gammas_singles = betas[::-1]
        gammas_pairs = betas[::-1]

        params = cls(qaoa_descriptor, betas, gammas_singles, gammas_pairs)

        return params

    @classmethod
    def random(cls, qaoa_descriptor: QAOADescriptor, seed: int = None):
        """
        Returns
        -------
        StandardParams
            Randomly initialised ``StandardParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        betas = np.random.uniform(0, np.pi, qaoa_descriptor.p)
        gammas_singles = np.random.uniform(0, np.pi, qaoa_descriptor.p)
        gammas_pairs = np.random.uniform(0, np.pi, qaoa_descriptor.p)

        params = cls(qaoa_descriptor, betas, gammas_singles, gammas_pairs)
        return params

    @classmethod
    def empty(cls, qaoa_descriptor: QAOADescriptor):
        """
        Initialise Standard Variational params with empty arrays
        """
        p = qaoa_descriptor.p
        betas = np.empty(p)
        gammas_singles = np.empty(p)
        gammas_pairs = np.empty(p)

        return cls(qaoa_descriptor, betas, gammas_singles, gammas_pairs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_singles):
            ax.plot(
                self.gammas_singles, label="gammas_singles", marker="^", ls="", **kwargs
            )
        if not _is_iterable_empty(self.gammas_pairs):
            ax.plot(
                self.gammas_pairs, label="gammas_pairs", marker="v", ls="", **kwargs
            )
        ax.set_xlabel("p")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.grid(linestyle='--')
        ax.legend()


# EXTENDED PARAMS ###################################


class QAOAVariationalExtendedParams(QAOAVariationalBaseParams):
    """
    QAOA parameters in their most general form with different angles for each
    operator.

    This means, that at the i-th timestep the evolution hamiltonian is given by

    .. math::

        H(t_i) = \sum_{\textrm{qubits } j} \beta_{ij} X_j
               + \sum_{\textrm{qubits } j} \gamma_{\textrm{single } ij} Z_j
               + \sum_{\textrm{qubit pairs} (jk)} \gamma_{\textrm{pair } i(jk)} Z_j Z_k

    and the complete circuit is then

    .. math::

        U = e^{-i H(t_p)} \cdots e^{-iH(t_1)}.

    Attributes
    ----------
    qaoa_descriptor: QAOADescriptor
                Specify the circuit parameters to construct circuit angles to be
                used for training
    betas_singles: list
        2D array with the gammas from above for each timestep and qubit.
        1st index goes over the timelayers, 2nd over the qubits.
    betas_pairs : list
    gammas_pairs: list
    gammas_singles: list
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        betas_singles: List[Union[float, int]],
        betas_pairs: List[Union[float, int]],
        gammas_singles: List[Union[float, int]],
        gammas_pairs: List[Union[float, int]],
    ):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(qaoa_descriptor)

        self.betas_singles = betas_singles if self.mixer_1q_coeffs else []
        self.betas_pairs = betas_pairs if self.mixer_2q_coeffs else []
        self.gammas_singles = gammas_singles if self.cost_1q_coeffs else []
        self.gammas_pairs = gammas_pairs if self.cost_2q_coeffs else []

    def __repr__(self):
        string = "Extended Parameterisation:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "Parameters:\n"
        string += (
            "\tbetas_singles: " + str(self.betas_singles).replace("\n", ",") + "\n"
        )
        string += "\tbetas_pairs: " + str(self.betas_pairs).replace("\n", ",") + "\n"
        string += (
            "\tgammas_singles: " + str(self.gammas_singles).replace("\n", ",") + "\n"
        )
        string += "\tgammas_pairs: " + str(self.gammas_pairs).replace("\n", ",") + "\n"
        return string

    def __len__(self):
        return self.p * (
            len(self.mixer_1q_coeffs)
            + len(self.mixer_2q_coeffs)
            + len(self.cost_1q_coeffs)
            + len(self.cost_2q_coeffs)
        )

    @shapedArray
    def betas_singles(self):
        return (self.p, len(self.mixer_1q_coeffs))

    @shapedArray
    def betas_pairs(self):
        return (self.p, len(self.mixer_2q_coeffs))

    @shapedArray
    def gammas_singles(self):
        return (self.p, len(self.cost_1q_coeffs))

    @shapedArray
    def gammas_pairs(self):
        return (self.p, len(self.cost_2q_coeffs))

    @property
    def mixer_1q_angles(self):
        return 2 * (self.mixer_1q_coeffs * self.betas_singles)

    @property
    def mixer_2q_angles(self):
        return 2 * (self.mixer_2q_coeffs * self.betas_pairs)

    @property
    def cost_1q_angles(self):
        return 2 * (self.cost_1q_coeffs * self.gammas_singles)

    @property
    def cost_2q_angles(self):
        return 2 * (self.cost_2q_coeffs * self.gammas_pairs)

    def update_from_raw(self, new_values):
        self.betas_singles = np.array(new_values[: len(self.mixer_1q_coeffs) * self.p])
        self.betas_singles = self.betas_singles.reshape(
            (self.p, len(self.mixer_1q_coeffs))
        )

        new_values = new_values[len(self.betas_singles.flatten()) :]

        self.betas_pairs = np.array(new_values[: len(self.mixer_2q_coeffs) * self.p])
        self.betas_pairs = self.betas_pairs.reshape((self.p, len(self.mixer_2q_coeffs)))

        new_values = new_values[len(self.betas_pairs.flatten()) :]

        self.gammas_singles = np.array(new_values[: len(self.cost_1q_coeffs) * self.p])
        self.gammas_singles = self.gammas_singles.reshape(
            (self.p, len(self.cost_1q_coeffs))
        )

        new_values = new_values[len(self.gammas_singles.flatten()) :]

        self.gammas_pairs = np.array(new_values[: len(self.cost_2q_coeffs) * self.p])
        self.gammas_pairs = self.gammas_pairs.reshape(
            (self.p, len(self.cost_2q_coeffs))
        )

        new_values = new_values[len(self.gammas_pairs.flatten()) :]
        # PEP8 complains, but new_values could be np.array and not list!
        if len(new_values) != 0:
            raise RuntimeWarning(
                "Incorrect dimension specified for new_values"
                "to construct the new betas and new gammas"
            )

    def raw(self):
        raw_data = np.concatenate(
            (
                self.betas_singles.flatten(),
                self.betas_pairs.flatten(),
                self.gammas_singles.flatten(),
                self.gammas_pairs.flatten(),
            )
        )
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(
        cls, qaoa_descriptor: QAOADescriptor, time: float = None
    ):
        """

        Returns
        -------
        ExtendedParams
            The initial parameters according to a linear ramp for the Hamiltonian specified by
            register, terms, weights.

        Todo
        ----
        Refactor this s.t. it supers from __init__
        """
        # create evenly spaced timelayers at the centers of p intervals
        p = qaoa_descriptor.p
        if time is None:
            time = float(0.7 * p)

        dt = time / p

        n_gamma_singles = len(qaoa_descriptor.cost_single_qubit_coeffs)
        n_gamma_pairs = len(qaoa_descriptor.cost_pair_qubit_coeffs)
        n_beta_singles = len(qaoa_descriptor.mixer_single_qubit_coeffs)
        n_beta_pairs = len(qaoa_descriptor.mixer_pair_qubit_coeffs)

        betas = np.linspace(
            (dt / time) * (time * (1 - 0.5 / p)), (dt / time) * (time * 0.5 / p), p
        )
        gammas = betas[::-1]

        betas_singles = betas.repeat(n_beta_singles).reshape(p, n_beta_singles)
        betas_pairs = betas.repeat(n_beta_pairs).reshape(p, n_beta_pairs)
        gammas_singles = gammas.repeat(n_gamma_singles).reshape(p, n_gamma_singles)
        gammas_pairs = gammas.repeat(n_gamma_pairs).reshape(p, n_gamma_pairs)

        params = cls(
            qaoa_descriptor, betas_singles, betas_pairs, gammas_singles, gammas_pairs
        )
        return params

    @classmethod
    def random(cls, qaoa_descriptor: QAOADescriptor, seed: int = None):
        """
        Returns
        -------
        ExtendedParams
            Randomly initialised ``ExtendedParams`` object
        """
        if seed is not None:
            np.random.seed(seed)

        p = qaoa_descriptor.p
        n_gamma_singles = len(qaoa_descriptor.cost_single_qubit_coeffs)
        n_gamma_pairs = len(qaoa_descriptor.cost_pair_qubit_coeffs)
        n_beta_singles = len(qaoa_descriptor.mixer_single_qubit_coeffs)
        n_beta_pairs = len(qaoa_descriptor.mixer_pair_qubit_coeffs)

        betas_singles = np.random.uniform(0, np.pi, (p, n_beta_singles))
        betas_pairs = np.random.uniform(0, np.pi, (p, n_beta_pairs))
        gammas_singles = np.random.uniform(0, np.pi, (p, n_gamma_singles))
        gammas_pairs = np.random.uniform(0, np.pi, (p, n_gamma_pairs))

        params = cls(
            qaoa_descriptor, betas_singles, betas_pairs, gammas_singles, gammas_pairs
        )
        return params

    @classmethod
    def empty(cls, qaoa_descriptor: QAOADescriptor):
        """
        Initialise Extended parameters with empty arrays
        """

        p = qaoa_descriptor.p
        n_gamma_singles = len(qaoa_descriptor.cost_single_qubit_coeffs)
        n_gamma_pairs = len(qaoa_descriptor.cost_pair_qubit_coeffs)
        n_beta_singles = len(qaoa_descriptor.mixer_single_qubit_coeffs)
        n_beta_pairs = len(qaoa_descriptor.mixer_pair_qubit_coeffs)

        betas_singles = np.empty((p, n_beta_singles))
        betas_pairs = np.empty((p, n_beta_pairs))
        gammas_singles = np.empty((p, n_gamma_singles))
        gammas_pairs = np.empty((p, n_gamma_pairs))

        params = cls(
            qaoa_descriptor, betas_singles, betas_pairs, gammas_singles, gammas_pairs
        )
        return params

    def get_constraints(self):
        """Constraints on the parameters for constrained parameters.

        Returns
        -------
        List[Tuple]:
            A list of tuples (0, upper_boundary) of constraints on the
            parameters s.t. we are exploiting the periodicity of the cost
            function. Useful for constrained optimizers.

        """
        beta_constraints = [(0, math.pi)] * (
            len(self.betas_singles.flatten() + len(self.betas_pairs.flatten()))
        )

        beta_pair_constraints = [(0, math.pi / w) for w in self.mixer_2q_coeffs]
        beta_pair_constraints *= self.p

        beta_single_constraints = [(0, math.pi / w) for w in self.mixer_1q_coeffs]
        beta_single_constraints *= self.p

        gamma_pair_constraints = [(0, 2 * math.pi / w) for w in self.cost_2q_coeffs]
        gamma_pair_constraints *= self.p

        gamma_single_constraints = [(0, 2 * math.pi / w) for w in self.cost_1q_coeffs]
        gamma_single_constraints *= self.p

        all_constraints = (
            beta_single_constraints
            + beta_pair_constraints
            + gamma_single_constraints
            + gamma_pair_constraints
        )

        return all_constraints

    def plot(self, ax=None, **kwargs):
        list_names_ = ["betas singles", "betas pairs", "gammas singles", "gammas pairs"]
        list_values_ = [
            self.betas_singles % (2 * (np.pi)),
            self.betas_pairs % (2 * (np.pi)),
            self.gammas_singles % (2 * (np.pi)),
            self.gammas_pairs % (2 * (np.pi)),
        ]

        list_names, list_values = list_names_.copy(), list_values_.copy()

        n_pop = 0
        for i in range(len(list_values_)):
            if list_values_[i].size == 0:
                list_values.pop(i - n_pop)
                list_names.pop(i - n_pop)
                n_pop += 1

        n = len(list_values)
        p = self.p

        if ax is None:
            fig, ax = plt.subplots((n + 1) // 2, 2, figsize=(9, 9 if n > 2 else 5))

        fig.tight_layout(pad=4.0)

        for k, (name, values) in enumerate(zip(list_names, list_values)):
            i, j = k // 2, k % 2
            axes = ax[i, j] if n > 2 else ax[k]

            if values.size == p:
                axes.plot(values.T[0], marker="^", color="green", ls="", **kwargs)
                axes.set_xlabel("p", fontsize=12)
                axes.set_title(name)
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))

            else:
                n_terms = values.shape[1]
                plt1 = axes.pcolor(
                    np.arange(p),
                    np.arange(n_terms),
                    values.T,
                    vmin=0,
                    vmax=2 * np.pi,
                    cmap="seismic",
                )
                axes.set_aspect(p / n_terms)
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                axes.yaxis.set_major_locator(MaxNLocator(integer=True))
                axes.set_ylabel("terms")
                axes.set_xlabel("p")
                axes.set_title(name)

                plt.colorbar(plt1, **kwargs)

        if k == 0:
            ax[1].axis("off")
        elif k == 2:
            ax[1, 1].axis("off")


# add other params and params factor later


# COST FUNCTION ###################################


def expectation_value_classical(counts: Dict, hamiltonian: Hamiltonian):
    """
    Evaluate the cost function, i.e. expectation value ``$$\langle|H \rangle$$``
    w.r.t to the measurements results ``counts``.

    Parameters
    ----------
    counts: dict
                The counts of the measurements.
    hamiltonian: Hamiltonian
                The Cost Hamiltonian defined for the optimization problem
    """
    shots = sum(counts.values())
    cost = 0
    for basis_state, count in counts.items():
        cost += count * (bitstring_energy(hamiltonian, basis_state) / shots)
    return cost


def cvar_expectation_value_classical(
    counts: Dict, hamiltonian: Hamiltonian, alpha: float
):
    """
    CVaR computation of cost function. For the definition of the cost function, refer
    to https://arxiv.org/abs/1907.04769.

    Parameters
    ----------
    counts: `dict`
            The counts of the measurements.
    hamiltonian: `Hamiltonian`
            The Cost Hamiltonian defined for the optimization problem
    alpha: `float`
            The CVaR parameter.
    """
    assert alpha > 0 and alpha < 1, "Please specify a valid alpha value between 0 and 1"
    shots = sum(counts.values())

    cost_list = []

    # eigen-energy computation of each basis state in counts
    for basis_state, count in counts.items():
        cost_list.append(count * (bitstring_energy(hamiltonian, basis_state) / shots))

    # sort costs in ascending order
    sorted_cost_list = np.sort(np.array(cost_list))

    K = max([int(len(cost_list) * alpha), 1])
    cost = sum(sorted_cost_list[:K])

    return cost


def cost_function(counts: Dict, hamiltonian: Hamiltonian, alpha: float = 1):
    """
    The cost function to be used for QAOA training.

    Parameters
    ----------
    counts: `dict`
            The counts of the measurements.
    hamiltonian: `Hamiltonian`
            The Cost Hamiltonian defined for the optimization problem
    alpha: `float`
            The CVaR parameter.
    """
    if alpha == 1:
        return expectation_value_classical(counts, hamiltonian)
    else:
        return cvar_expectation_value_classical(counts, hamiltonian, alpha)


# BASEBACKENDS ######################################


class QuantumCircuitBase:
    """
    Phantom class to indicate Quantum Circuits constructed using
    several acceptable services. For instance, IBMQ, PyQuil
    """

    pass


class VQABaseBackend(ABC):
    """
    This is the Abstract Base Class over which other classes will be built.
    Since, this is an Abstract Base class, in order to prevent its initialisation
    the class methods -- ``__init__`` and ``__cal__`` will be decorated as
    `abstractmethods`.

    The Child classes MUST implement and override these abstract methods in their
    implementation specific to their needs.

    NOTE:
        In addition one can also implement other methods which are not
        necessitated by the ``VQABaseBackend`` Base Class


    Parameters
    ----------
    prepend_state: `Union[QuantumCircuitBase, List[complex], np.ndarray]`
        The initial state to start the quantum circuit in the backend.
    append_state: `Union[QuantumCircuitBase, np.ndarray]`
        The final state to append to the quantum circuit in the backend.
    """

    @abstractmethod
    def __init__(
        self,
        prepend_state: Optional[Union[QuantumCircuitBase, List[complex], np.ndarray]],
        append_state: Optional[Union[QuantumCircuitBase, np.ndarray]],
    ):
        """The constructor. See class docstring"""
        self.prepend_state = prepend_state
        self.append_state = append_state

    @abstractmethod
    def expectation(self, params: Any) -> float:
        """
        Call the execute function on the circuit to compute the
        expectation value of the Quantum Circuit w.r.t cost operator
        """
        pass

    @abstractmethod
    def expectation_w_uncertainty(self, params: Any) -> Tuple[float, float]:
        """
        Call the execute function on the circuit to compute the
        expectation value of the Quantum Circuit w.r.t cost operator
        along with its uncertainty
        """
        pass

    @abstractproperty
    def exact_solution(self):
        """
        Use linear algebra to compute the exact solution of the problem
        Hamiltonian classically.
        """
        pass


class QAOABaseBackend(VQABaseBackend):
    """
    This class inherits from the VQABaseBackend and needs to be backend
    agnostic to QAOA implementations on different devices and their
    respective SDKs.

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        This object handles the information to design the QAOA circuit ansatz
    prepend_state: `Union[QuantumCircuitBase, List[complex]]`
        Warm Starting the QAOA problem with some initial state other than the regular
        $|+ \\rangle ^{otimes n}$
    append_state: `Union[QuantumCircuitBase, List[complex]]`
        Appending a user-defined circuit/state to the end of the QAOA routine
    init_hadamard: `bool`
        Initialises the QAOA circuit with a hadamard when ``True``
    cvar_alpha: `float`
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        prepend_state: Optional[Union[QuantumCircuitBase, List[complex], np.ndarray]],
        append_state: Optional[Union[QuantumCircuitBase, np.ndarray]],
        init_hadamard: bool,
        cvar_alpha: float,
    ):
        super().__init__(prepend_state, append_state)

        self.qaoa_descriptor = qaoa_descriptor
        self.cost_hamiltonian = qaoa_descriptor.cost_hamiltonian
        self.n_qubits = self.qaoa_descriptor.n_qubits
        self.init_hadamard = init_hadamard
        self.cvar_alpha = cvar_alpha
        self.problem_qubits = self.qaoa_descriptor.cost_hamiltonian.n_qubits

        self.abstract_circuit = deepcopy(self.qaoa_descriptor.abstract_circuit)

        # pass the generated mappings if the circuit is routed
        if self.qaoa_descriptor.routed == True:
            self.initial_qubit_mapping = self.qaoa_descriptor.initial_mapping

            if self.qaoa_descriptor.p % 2 != 0:
                self.final_mapping = self.qaoa_descriptor.final_mapping
            else:
                # if even, the initial mapping [0,...,n_qubits-1] is taken as the final mapping
                self.final_mapping = list(
                    range(len(self.qaoa_descriptor.final_mapping))
                )
        else:
            self.initial_qubit_mapping = None
            self.final_mapping = None

    def assign_angles(self, params: QAOAVariationalBaseParams) -> None:
        """
        Assigns the angle values of the variational parameters to the circuit gates
        specified as a list of gates in the ``abstract_circuit``.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The variational parameters(angles) to be assigned to the circuit gates
        """
        # if circuit is non-parameterised, then assign the angle values to the circuit
        abstract_circuit = self.abstract_circuit

        for each_gate in abstract_circuit:
            gate_label_layer = each_gate.gate_label.layer
            gate_label_seq = each_gate.gate_label.sequence
            if each_gate.gate_label.n_qubits == 2:
                if each_gate.gate_label.type.value == "MIXER":
                    angle = params.mixer_2q_angles[gate_label_layer, gate_label_seq]
                elif each_gate.gate_label.type.value == "COST":
                    angle = params.cost_2q_angles[gate_label_layer, gate_label_seq]
            elif each_gate.gate_label.n_qubits == 1:
                if each_gate.gate_label.type.value == "MIXER":
                    angle = params.mixer_1q_angles[gate_label_layer, gate_label_seq]
                elif each_gate.gate_label.type.value == "COST":
                    angle = params.cost_1q_angles[gate_label_layer, gate_label_seq]
            each_gate.angle_value = angle

        self.abstract_circuit = abstract_circuit

    def obtain_angles_for_pauli_list(
        self, input_gate_list: List[GateMap], params: QAOAVariationalBaseParams
    ) -> List[float]:
        """
        This method uses the pauli gate list information to obtain the pauli angles
        from the VariationalBaseParams object. The floats in the list are in the order
        of the input GateMaps list.

        Parameters
        ----------
        input_gate_list: `List[GateMap]`
            The GateMap list including rotation gates
        params: `QAOAVariationalBaseParams`
            The variational parameters(angles) to be assigned to the circuit gates

        Returns
        -------
        angles_list: `List[float]`
            The list of angles in the order of gates in the `GateMap` list
        """
        angle_list = []

        for each_gate in input_gate_list:
            gate_label_layer = each_gate.gate_label.layer
            gate_label_seq = each_gate.gate_label.sequence

            if each_gate.gate_label.n_qubits == 2:
                if each_gate.gate_label.type.value == "MIXER":
                    angle_list.append(
                        params.mixer_2q_angles[gate_label_layer, gate_label_seq]
                    )
                elif each_gate.gate_label.type.value == "COST":
                    angle_list.append(
                        params.cost_2q_angles[gate_label_layer, gate_label_seq]
                    )
            elif each_gate.gate_label.n_qubits == 1:
                if each_gate.gate_label.type.value == "MIXER":
                    angle_list.append(
                        params.mixer_1q_angles[gate_label_layer, gate_label_seq]
                    )
                elif each_gate.gate_label.type.value == "COST":
                    angle_list.append(
                        params.cost_1q_angles[gate_label_layer, gate_label_seq]
                    )

        return angle_list

    @abstractmethod
    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> QuantumCircuitBase:
        """
        Construct the QAOA circuit and append the parameter values to obtain the final
        circuit ready for execution on the device.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of
            the parameter classes, containing hyperparameters and variable parameters).

        Returns
        -------
        quantum_circuit: `QuantumCircuitBase`
            A Quantum Circuit object of type created by the respective
            backend service
        """
        pass

    @abstractmethod
    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        This method will be implemented in the child classes according to the type
        of backend used.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: `int`
            The number of shots to be used for the measurement. If None, the backend default.
        """
        pass

    @round_value
    def expectation(self, params: QAOAVariationalBaseParams, n_shots=None) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: `int`
            The number of shots to be used for the measurement. If None, the backend default.

        Returns
        -------
        float:
            Expectation value of cost operator wrt to quantum state produced by QAOA circuit
        """
        counts = self.get_counts(params, n_shots)
        cost = cost_function(
            counts, self.qaoa_descriptor.cost_hamiltonian, self.cvar_alpha
        )
        return cost

    @round_value
    def expectation_w_uncertainty(
        self, params: QAOAVariationalBaseParams, n_shots=None
    ) -> Tuple[float, float]:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian and its uncertainty

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: `int`
            The number of shots to be used for the measurement. If None, the backend default.

        Returns
        -------
        Tuple[float]:
            expectation value and its uncertainty of cost operator wrt
            to quantum state produced by QAOA circuit.
        """
        counts = self.get_counts(params, n_shots)
        cost = cost_function(
            counts, self.qaoa_descriptor.cost_hamiltonian, self.cvar_alpha
        )
        cost_sq = cost_function(
            counts,
            self.qaoa_descriptor.cost_hamiltonian.hamiltonian_squared,
            self.cvar_alpha,
        )

        uncertainty = np.sqrt(cost_sq - cost**2)

        return (cost, uncertainty)

    @abstractmethod
    def reset_circuit(self):
        """
        Reset the circuit attribute
        """
        pass

    @property
    def exact_solution(self):
        """
        Computes exactly the minimum energy of the cost function and its
        corresponding configuration of variables using standard numpy module.

        Returns
        -------
        (energy, config): `Tuple[float, list]`
            - The minimum eigenvalue of the cost Hamiltonian,
            - The minimum energy eigenvector as a binary array
              configuration: qubit-0 as the first element in the sequence
        """
        register = self.qaoa_descriptor.qureg
        terms = self.cost_hamiltonian.terms
        coeffs = self.cost_hamiltonian.coeffs
        constant_energy = self.cost_hamiltonian.constant

        diag = np.zeros((2 ** len(register)))
        for i, term in enumerate(terms):
            out = np.real(coeffs[i])
            for qubit in register:
                if qubit in term.qubit_indices:
                    out = np.kron([1, -1], out)
                else:
                    out = np.kron([1, 1], out)
            diag += out

        # add the constant energy contribution
        diag += constant_energy

        # index = np.argmin(diag)
        energy = np.min(diag)
        indices = []
        for idx in range(len(diag)):
            if diag[idx] == energy:
                indices.append(idx)

        config_strings = [
            np.binary_repr(index, len(register))[::-1] for index in indices
        ]
        configs = [
            np.array([int(x) for x in config_str]) for config_str in config_strings
        ]

        return energy, configs

    def bitstring_energy(self, bitstring: Union[List[int], str]) -> float:
        """
        Computes the energy of a given bitstring with respect to the cost Hamiltonian.

        Parameters
        ----------
        bitstring : `Union[List[int],str]`
            A list of integers 0 and 1 of length `n_qubits` representing a configuration.

        Returns
        -------
        float:
            The energy of a given bitstring with respect to the cost Hamiltonian.
        """
        energy = 0
        string_rev = bitstring
        terms = self.cost_hamiltonian.terms
        coeffs = self.cost_hamiltonian.coeffs
        constant_energy = self.cost_hamiltonian.constant

        for i, term in enumerate(terms):
            variables_product = np.prod([(-1) ** string_rev[k] for k in term])
            energy += coeffs[i] * variables_product
        energy += constant_energy

        return energy

    @abstractmethod
    def circuit_to_qasm(self):
        """
        Implement a method to construct a QASM string from the current
        state of the QuantumCircuit for the backends
        """
        pass


class QAOABaseBackendShotBased(QAOABaseBackend):
    """
    Implementation of Backend object specific to shot-based simulators and QPUs
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        n_shots: int,
        prepend_state: Optional[QuantumCircuitBase],
        append_state: Optional[QuantumCircuitBase],
        init_hadamard: bool,
        cvar_alpha: float,
    ):
        super().__init__(
            qaoa_descriptor, prepend_state, append_state, init_hadamard, cvar_alpha
        )
        # assert self.n_qubits >= len(prepend_state.qubits), \
        # "Cannot attach a bigger circuit to the QAOA routine"
        # assert self.n_qubits >= len(append_state.qubits), \
        # "Cannot attach a bigger circuit to the QAOA routine"
        self.n_shots = n_shots

    @abstractmethod
    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Measurement outcome vs frequency information from a circuit execution
        represented as a python dictionary

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters as a 1D array (derived from an object of one of the
            parameter classes, containing hyperparameters and variable parameters).
        n_shots: `int`
            The number of shots to be used for the measurement. If None, the backend default.

        Returns
        -------
        Dict[str, float]:
            A dictionary of measurement outcomes vs frequency sampled from a statevector
        """
        pass


class QAOABaseBackendCloud:
    """
    QAOA backend that can be accessed over the cloud offered by the
    respective provider through an API based access
    """

    def __init__(self, device):
        self.device = device


class QAOABaseBackendParametric:
    """
    Base class to indicate Parametric Circuit Backend
    """

    @abstractmethod
    def parametric_qaoa_circuit(self):
        pass


# QISKIT DEVICE OQ ##################################


class DeviceQiskit:
    """
    Contains the required information and methods needed to access remote
    qiskit QPUs.

    Attributes
    ----------
    available_qpus: `list`
      When connection to a provider is established, this attribute contains a list
      of backend names which can be used to access the selected backend by reinitialising
      the Access Object with the name of the available backend as input to the
      device_name parameter.
    n_qubits: `int`
        The maximum number of qubits available for the selected backend. Only
        available if check_connection method is executed and a connection to the
        qpu and provider is established.
    """

    def __init__(
        self,
        ibmq_backend,
    ):
        """The user's IBMQ account has to be authenticated through qiskit in
        order to use this backend. This can be done through `IBMQ.save_account`.

        See: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq

        Parameters
        ----------
        ibmq_backend: IBMQBackend
        """

        self.device_name = ibmq_backend.name
        self.device_location = "ibmq"
        self.backend_device = ibmq_backend
        # self.hub = hub
        # self.group = group
        # self.project = project
        # self.as_emulator = as_emulator

        self.provider_connected = True
        self.qpu_connected = True

    def check_connection(self) -> bool:
        """
        This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        If no backend was specified in initialisation of object, just runs
        a test connection without a specific backend.
        If backend was specified, checks if connection to that backend
        can be established.

        Returns
        -------
        bool
                        True if successfully connected to IBMQ or IBMQ and the QPU backend
                        if it was specified. False if unable to connect to IBMQ or failure
                        in the attempt to connect to the specified backend.
        """

        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False

    def _check_backend_connection(self) -> bool:
        """Private method for checking connection with backend(s)."""
        self.n_qubits = self.backend_device.configuration().n_qubits
        return True
    
    def _check_provider_connection(self) -> bool:
        return True

    def connectivity(self) -> List[List[int]]:
        return self.backend_device.configuration().coupling_map


# QISKIT BACKEND TO IMPLEMENT AND EXECUTE THE CIRCUIT ##########


class QAOAQiskitQPUBackend(
    QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased
):
    """
    A QAOA simulator as well as for real QPU using qiskit as the backend

    Parameters
    ----------
    device: `DeviceQiskit`
        An object of the class ``DeviceQiskit`` which contains the credentials
        for accessing the QPU via cloud and the name of the device.
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.
    n_shots: `int`
        The number of shots to be taken for each circuit.
    prepend_state: `QuantumCircuit`
        The state prepended to the circuit.
    append_state: `QuantumCircuit`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.
    cvar_alpha: `float`
        The value of alpha for the CVaR method.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        device,
        n_shots: int,
        prepend_state: Optional[QuantumCircuit],
        append_state: Optional[QuantumCircuit],
        init_hadamard: bool,
        initial_qubit_mapping: Optional[List[int]] = None,
        qiskit_optimization_level: int = 1,
        cvar_alpha: float = 1,
    ):
        QAOABaseBackendShotBased.__init__(
            self,
            qaoa_descriptor,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )
        QAOABaseBackendCloud.__init__(self, device)

        self.qureg = QuantumRegister(self.n_qubits)
        self.problem_reg = self.qureg[0 : self.problem_qubits]
        self.creg = ClassicalRegister(len(self.problem_reg))

        if qiskit_optimization_level in [0, 1, 2, 3]:
            self.qiskit_optimziation_level = qiskit_optimization_level
        else:
            raise ValueError(
                f"qiskit_optimization_level cannot be {qiskit_optimization_level}. Choose between 0 to 3"
            )
        self.gate_applicator = QiskitGateApplicator()

        if self.initial_qubit_mapping is None:
            self.initial_qubit_mapping = (
                initial_qubit_mapping
                if initial_qubit_mapping is not None
                else list(range(self.n_qubits))
            )

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), (
                "Cannot attach a bigger circuit" "to the QAOA routine"
            )

        if self.device.provider_connected and self.device.qpu_connected:
            self.backend_qpu = self.device.backend_device
        elif self.device.provider_connected and self.device.qpu_connected in [
            False,
            None,
        ]:
            raise Exception(
                "Connection to {} was made. Error connecting to the specified backend.".format(
                    self.device.device_location.upper()
                )
            )

        else:
            raise Exception(
                "Error connecting to {}.".format(self.device.device_location.upper())
            )

        if self.device.n_qubits < self.n_qubits:
            raise Exception(
                "There are lesser qubits on the device than the number of qubits required for the circuit."
            )
        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> QuantumCircuit:
        """
        The final QAOA circuit to be executed on the QPU.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        qaoa_circuit: `QuantumCircuit`
            The final QAOA circuit after binding angles from variational parameters.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        memory_map = dict(zip(self.qiskit_parameter_list, angles_list))
        circuit_with_angles = self.parametric_circuit.bind_parameters(memory_map)

        if self.append_state:
            circuit_with_angles = circuit_with_angles.compose(self.append_state)

        # only measure the problem qubits
        if self.final_mapping is None:
            circuit_with_angles.measure(self.problem_reg, self.creg)
        else:
            for idx, qubit in enumerate(self.final_mapping[0 : len(self.problem_reg)]):
                cbit = self.creg[idx]
                circuit_with_angles.measure(qubit, cbit)

        if self.qaoa_descriptor.routed is True:
            transpiled_circuit = transpile(
                circuit_with_angles,
                self.backend_qpu,
                initial_layout=self.initial_qubit_mapping,
                optimization_level=self.qiskit_optimziation_level,
                routing_method="none",
            )
        else:
            transpiled_circuit = transpile(
                circuit_with_angles,
                self.backend_qpu,
                initial_layout=self.initial_qubit_mapping,
                optimization_level=self.qiskit_optimziation_level,
            )

        return transpiled_circuit

    @property
    def parametric_qaoa_circuit(self) -> QuantumCircuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
            params:
                Object of type QAOAVariationalBaseParams
        """
        # self.reset_circuit()
        parametric_circuit = QuantumCircuit(self.qureg, self.creg)

        if self.prepend_state:
            parametric_circuit = parametric_circuit.compose(self.prepend_state)
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.h(self.problem_reg)

        self.qiskit_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = Parameter(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            decomposition = each_gate.decomposition("standard")
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Execute the circuit and obtain the counts

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: int
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots

        Returns
        -------
            A dictionary with the bitstring as the key and the number of counts
            as its value.
        """

        n_shots = self.n_shots if n_shots is None else n_shots

        circuit = self.qaoa_circuit(params)

        job_state = False
        no_of_job_retries = 0
        max_job_retries = 5

        job = self.backend_qpu.run(circuit, shots=n_shots)
        self.job_id = job.job_id()

        counts = job.result().get_counts()
        # Expose counts
        final_counts = flip_counts(counts)
        self.measurement_outcomes = final_counts
        return final_counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the entire QAOA `QuantumCircuit` object into
        a OpenQASM string
        """
        raise NotImplementedError()
        # qasm_string = self.qaoa_circuit(params).qasm(formatted=True)
        # return qasm_string

    def reset_circuit(self):
        """
        Reset self.circuit after performing a computation
        """
        raise NotImplementedError()


# QISKIT GATES ######################################


class QiskitGateApplicator(GateApplicator):
    QISKIT_OQ_GATE_MAPPER = {
        X.__name__: XGate,
        RZ.__name__: RZGate,
        RX.__name__: RXGate,
        RY.__name__: RYGate,
        CX.__name__: CXGate,
        CZ.__name__: CZGate,
        RXX.__name__: RXXGate,
        RZX.__name__: RZXGate,
        RZZ.__name__: RZZGate,
        RYY.__name__: RYYGate,
        CPHASE.__name__: CRZGate,
    }

    library = "qiskit"

    def create_quantum_circuit(self, n_qubits) -> QuantumCircuit:
        """
        Function which creates and empty circuit specific to the qiskit backend.
        Needed for SPAM twirling but more general than this.
        """
        qureg = QuantumRegister(n_qubits)
        parametric_circuit = QuantumCircuit(qureg)
        return parametric_circuit

    def gate_selector(self, gate: Gate) -> Callable:
        selected_qiskit_gate = QiskitGateApplicator.QISKIT_OQ_GATE_MAPPER[gate.__name__]
        return selected_qiskit_gate

    @staticmethod
    def apply_1q_rotation_gate(
        qiskit_gate,
        qubit_1: int,
        rotation_object: RotationAngle,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        circuit.append(qiskit_gate(rotation_object.rotation_angle), [qubit_1], [])
        return circuit

    @staticmethod
    def apply_2q_rotation_gate(
        qiskit_gate,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
        circuit: QuantumCircuit,
    ) -> QuantumCircuit:
        circuit.append(
            qiskit_gate(rotation_object.rotation_angle), [qubit_1, qubit_2], []
        )
        return circuit

    @staticmethod
    def apply_1q_fixed_gate(
        qiskit_gate, qubit_1: int, circuit: QuantumCircuit
    ) -> QuantumCircuit:
        circuit.append(qiskit_gate(), [qubit_1], [])
        return circuit

    @staticmethod
    def apply_2q_fixed_gate(
        qiskit_gate, qubit_1: int, qubit_2: int, circuit: QuantumCircuit
    ) -> QuantumCircuit:
        circuit.append(qiskit_gate(), [qubit_1, qubit_2], [])
        return circuit

    def apply_gate(self, gate: Gate, *args):
        selected_qiskit_gate = self.gate_selector(gate)
        if gate.n_qubits == 1:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,rotation_object,circuit)
                return self.apply_1q_rotation_gate(selected_qiskit_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,circuit)
                return self.apply_1q_fixed_gate(selected_qiskit_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object,circuit)
                return self.apply_2q_rotation_gate(selected_qiskit_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2,circuit)
                return self.apply_2q_fixed_gate(selected_qiskit_gate, *args)
        else:
            raise ValueError("Only 1 and 2-qubit gates are supported.")


# OQ TRAINING_VQA #####################################


def save_parameter(parameter_name: str, parameter_value: Any):
    filename = "oq_saved_info_" + parameter_name

    try:
        opened_csv = pd.read_csv(filename + ".csv")
    except Exception:
        opened_csv = pd.DataFrame(columns=[parameter_name])

    if type(parameter_value) not in [str, float, int]:
        parameter_value = str(parameter_value)

    update_df = pd.DataFrame(data={parameter_name: parameter_value}, index=[0])
    new_df = pd.concat([opened_csv, update_df], ignore_index=True)

    new_df.to_csv(filename + ".csv", index=False)

    print("Parameter Saving Successful")


class OptimizeVQA(ABC):
    """
    Training Class for optimizing VQA algorithm that wraps around VQABaseBackend and QAOAVariationalBaseParams objects.
    This function utilizes the `update_from_raw` of the QAOAVariationalBaseParams class and `expectation` method of
    the VQABaseBackend class to create a wrapper callable which is passed into scipy.optimize.minimize for minimization.
    Only the trainable parameters should be passed instead of the complete
    AbstractParams object. The construction is completely backend and type
    of VQA agnostic.

    This class is an AbstractBaseClass on top of which other specific Optimizer
    classes are built.

    .. Tip::
        Optimizer that usually work the best for quantum optimization problems

        * Gradient free optimizer - Cobyla

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the
        backend used to perform computations, and the VQA circuit.
    variational_params:
        Object of class QAOAVariationalBaseParams, which contains information on the circuit to be executed,
        the type of parametrisation, and the angles of the VQA circuit.
    method:
        which method to use for optimization. Choose a method from the list
        of supported methods by scipy optimize, or from the list of custom gradient optimisers.
    optimizer_dict:
        All extra parameters needed for customising the optimising, as a dictionary.
    Optimizers that usually work the best for quantum optimization problems:
        #. Gradient free optimizer: BOBYQA, ImFil, Cobyla
        #. Gradient based optimizer: L-BFGS, ADAM (With parameter shift gradients)

        Note: Adam is not a part of scipy, it will added in a future version
    """

    def __init__(
        self,
        vqa_object: Type[VQABaseBackend],
        variational_params: Type[QAOAVariationalBaseParams],
        optimizer_dict: dict,
        publisher: Publisher,
    ):
        if not isinstance(vqa_object, VQABaseBackend):
            raise TypeError("The specified cost object must be of type VQABaseBackend")

        self.vqa = vqa_object
        # extract initial parameters from the params of vqa_object
        self.variational_params = variational_params
        self.initial_params = variational_params.raw()
        self.method = optimizer_dict["method"].lower()
        self.save_to_csv = optimizer_dict.get("save_intermediate", False)

        self.log = Logger(
            {
                "cost": {
                    "history_update_bool": optimizer_dict.get("cost_progress", True),
                    "best_update_string": "LowestOnly",
                },
                "measurement_outcomes": {
                    "history_update_bool": optimizer_dict.get(
                        "optimization_progress", False
                    ),
                    "best_update_string": "Replace",
                },
                "param_log": {
                    "history_update_bool": optimizer_dict.get("parameter_log", True),
                    "best_update_string": "Replace",
                },
                "eval_number": {
                    "history_update_bool": False,
                    "best_update_string": "Replace",
                },
                "func_evals": {
                    "history_update_bool": False,
                    "best_update_string": "HighestOnly",
                },
                "jac_func_evals": {
                    "history_update_bool": False,
                    "best_update_string": "HighestOnly",
                },
                "qfim_func_evals": {
                    "history_update_bool": False,
                    "best_update_string": "HighestOnly",
                },
                "job_ids": {
                    "history_update_bool": True,
                    "best_update_string": "Replace",
                },
                "n_shots": {
                    "history_update_bool": True,
                    "best_update_string": "Replace",
                },
            },
            {
                "root_nodes": [
                    "cost",
                    "func_evals",
                    "jac_func_evals",
                    "qfim_func_evals",
                    "n_shots",
                ],
                "best_update_structure": (
                    ["cost", "param_log"],
                    ["cost", "measurement_outcomes"],
                    ["cost", "job_ids"],
                    ["cost", "eval_number"],
                ),
            },
        )

        self.log.log_variables(
            {"func_evals": 0, "jac_func_evals": 0, "qfim_func_evals": 0}
        )
        self.publisher = publisher

    @abstractmethod
    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer.
        """
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()}\n"

        return string

    def __call__(self):
        """
        Call the class instance to initiate the training process.
        """
        self.optimize()
        return self

    # def evaluate_jac(self, x):

    def optimize_this(self, x, n_shots=None):
        """
        A function wrapper to execute the circuit in the backend. This function
        will be passed as argument to be optimized by scipy optimize.

        Parameters
        ----------
        x:
            Parameters (a list of floats) over which optimization is performed.
        n_shots:
            Number of shots to be used for the backend when computing the expectation.
            If None, nothing is passed to the backend.

        Returns
        -------
        cost value:
            Cost value which is evaluated on the declared backend.

        Returns
        -------
        :
            Cost Value evaluated on the declared backed or on the Wavefunction Simulator if specified so
        """

        log_dict = {}

        self.variational_params.update_from_raw(deepcopy(x))
        log_dict.update({"param_log": self.variational_params.raw()})

        if hasattr(self.vqa, "log_with_backend") and callable(
            getattr(self.vqa, "log_with_backend")
        ):
            self.vqa.log_with_backend(
                metric_name="variational_params",
                value=self.variational_params,
                iteration_number=self.log.func_evals.best[0],
            )

        if self.save_to_csv:
            save_parameter("param_log", deepcopy(x))

        n_shots_dict = {"n_shots": n_shots} if n_shots else {}
        callback_cost = self.vqa.expectation(self.variational_params, **n_shots_dict)

        log_dict.update({"cost": callback_cost})

        current_eval = self.log.func_evals.best[0]
        current_eval += 1
        log_dict.update({"func_evals": current_eval})

        log_dict.update(
            {
                "eval_number": (
                    current_eval
                    - self.log.jac_func_evals.best[0]
                    - self.log.qfim_func_evals.best[0]
                )
            }
        )  # this one will say which evaluation is the optimized one

        log_dict.update({"measurement_outcomes": self.vqa.measurement_outcomes})

        if hasattr(self.vqa, "log_with_backend") and callable(
            getattr(self.vqa, "log_with_backend")
        ):
            self.vqa.log_with_backend(
                metric_name="measurement_outcomes",
                value=self.vqa.measurement_outcomes,
                iteration_number=self.log.func_evals.best[0],
            )

        if hasattr(self.vqa, "job_id"):
            log_dict.update({"job_ids": self.vqa.job_id})

            if self.save_to_csv:
                save_parameter("job_ids", self.vqa.job_id)

        self.log.log_variables(log_dict)

        self.publisher.callback(
            callback_cost, self.vqa.job_id, self.vqa.measurement_outcomes
        )

        return callback_cost

    @abstractmethod
    def optimize(self):
        """
        Main method which implements the optimization process.
        Child classes must implement this method according to their respective
        optimization process.

        Returns
        -------
        :
            The optimized return object from the ``scipy.optimize`` package the result
            is assigned to the attribute ``opt_result``
        """
        pass

    def results_dictionary(self):
        """
        This method formats a dictionary that consists of all the results from
        the optimization process. The dictionary is returned by this method.
        The results can also be saved by providing the path to save the pickled file

        .. Important::
            Child classes must implement this method so that the returned object,
            a ``Dictionary`` is consistent across all Optimizers.

        TODO:
            Decide results datatype: dictionary or namedtuple?

        Parameters
        ----------
        file_path:
            To save the results locally on the machine in pickle format, specify
            the entire file path to save the result_dictionary.

        file_name:
            Custom name for to save the data; a generic name with the time of
            optimization is used if not specified
        """
        # date_time = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
        # file_name = f"opt_results_{date_time}" if file_name is None else file_name

        self.qaoa_result = QAOAResult(
            self.log, self.method, self.vqa.cost_hamiltonian, type(self.vqa)
        )

        # if file_path and os.path.isdir(file_path):
        #     print("Saving results locally")
        #     pickled_file = open(f"{file_path}/{file_name}.pcl", "wb")
        #     pickle.dump(self.qaoa_result, pickled_file)
        #     pickled_file.close()

        # return  #result_dict


class ScipyOptimizer(OptimizeVQA):
    """
    Python vanilla scipy based optimizer for the VQA class.

    .. Tip::
        Using bounds may result in lower optimization performance

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the
        backend used to perform computations, and the VQA circuit.

    variational_params:
        Object of class QAOAVariationalBaseParams, which contains information on
        the circuit to be executed,  the type of parametrisation,
        and the angles of the VQA circuit.

    optimizer_dict:
        * 'jac': gradient as ``Callable``, if defined else ``None``
        * 'hess': hessian as ``Callable``, if defined else ``None``
        * 'bounds': parameter bounds while training, defaults to ``None``
        * 'constraints': Linear/Non-Linear constraints (only for COBYLA, SLSQP and trust-constr)
        * 'tol': Tolerance for termination
        * 'maxiter': sets ``maxiters = 100`` by default if not specified.
        * 'maxfev': sets ``maxfev = 100`` by default if not specified.
        * 'optimizer_options': dictionary of optimiser-specific arguments, defaults to ``None``
    """

    GRADIENT_FREE = ["cobyla", "nelder-mead", "powell", "slsqp"]
    SCIPY_METHODS = MINIMIZE_METHODS

    def __init__(
        self,
        vqa_object: Type[VQABaseBackend],
        variational_params: Type[QAOAVariationalBaseParams],
        optimizer_dict: dict,
        publisher: Publisher
    ):
        super().__init__(vqa_object, variational_params, optimizer_dict, publisher)

        self.vqa_object = vqa_object
        self._validate_and_set_params(optimizer_dict)

    def _validate_and_set_params(self, optimizer_dict):
        """
        Verify that the specified arguments are valid for the particular optimizer.
        """

        if self.method not in ScipyOptimizer.SCIPY_METHODS:
            raise ValueError("Specified method not supported by Scipy Minimize")

        jac = optimizer_dict.get("jac", None)
        hess = optimizer_dict.get("hess", None)
        jac_options = optimizer_dict.get("jac_options", None)
        hess_options = optimizer_dict.get("hess_options", None)

        if self.method not in ScipyOptimizer.GRADIENT_FREE and (
            jac is None or not isinstance(jac, (Callable, str))
        ):
            raise ValueError(
                "Please specify either a string or provide callable"
                "gradient in order to use gradient based methods"
            )
        else:
            if isinstance(jac, str):
                self.jac = derivative(
                    self.vqa_object,
                    self.variational_params,
                    self.log,
                    "gradient",
                    jac,
                    jac_options,
                )
            else:
                self.jac = jac

        hess = optimizer_dict.get("hess", None)
        if hess is not None and not isinstance(hess, (Callable, str)):
            raise ValueError("Hessian needs to be of type Callable or str")
        else:
            if isinstance(hess, str):
                self.hess = derivative(
                    self.vqa_object,
                    self.variational_params,
                    self.log,
                    "hessian",
                    hess,
                    hess_options,
                )
            else:
                self.hess = hess

        constraints = optimizer_dict.get("constraints", ())
        if (
            constraints == ()
            or isinstance(constraints, LinearConstraint)
            or isinstance(constraints, NonlinearConstraint)
        ):
            self.constraints = constraints
        else:
            raise ValueError(
                f"Constraints for Scipy optimization should be of type {LinearConstraint} or {NonlinearConstraint}"
            )

        bounds = optimizer_dict.get("bounds", None)

        if bounds is None or isinstance(bounds, Bounds):
            self.bounds = bounds
        elif isinstance(bounds, List):
            lb = np.array(bounds).T[0]
            ub = np.array(bounds).T[1]
            self.bounds = Bounds(lb, ub)
        else:
            raise ValueError(
                f"Bounds for Scipy optimization should be of type {Bounds},"
                "or a list in the form [[ub1, lb1], [ub2, lb2], ...]"
            )

        self.options = optimizer_dict.get("optimizer_options", {})
        self.options["maxiter"] = optimizer_dict.get("maxiter", None)
        if optimizer_dict.get("maxfev") is not None:
            self.options["maxfev"] = optimizer_dict.get("maxfev", None)

        self.tol = optimizer_dict.get("tol", None)

        return self

    def __repr__(self):
        """
        Overview of the instantiated optimier/trainer.
        """
        maxiter = self.options["maxiter"]
        string = f"Optimizer for VQA of type: {type(self.vqa).__base__.__name__} \n"
        string += f"Backend: {type(self.vqa).__name__} \n"
        string += f"Method: {str(self.method).upper()} with Max Iterations: {maxiter}\n"

        return string

    def optimize(self):
        """
        Main method which implements the optimization process using ``scipy.minimize``.

        Returns
        -------
        :
            Returns self after the optimization process is completed.
        """

        try:
            if self.method not in ScipyOptimizer.GRADIENT_FREE:
                if self.hess == None:
                    result = minimize(
                        self.optimize_this,
                        x0=self.initial_params,
                        method=self.method,
                        jac=self.jac,
                        tol=self.tol,
                        constraints=self.constraints,
                        options=self.options,
                        bounds=self.bounds,
                    )
                else:
                    result = minimize(
                        self.optimize_this,
                        x0=self.initial_params,
                        method=self.method,
                        jac=self.jac,
                        hess=self.hess,
                        tol=self.tol,
                        constraints=self.constraints,
                        options=self.options,
                        bounds=self.bounds,
                    )
            else:
                result = minimize(
                    self.optimize_this,
                    x0=self.initial_params,
                    method=self.method,
                    tol=self.tol,
                    constraints=self.constraints,
                    options=self.options,
                    bounds=self.bounds,
                )
        except ConnectionError as e:
            print(e, "\n")
            print(
                "The optimization has been terminated early. Most likely due to a connection error."
                "You can retrieve results from the optimization runs that were completed"
                "through the .result method."
            )
        except Exception as e:
            raise e
        finally:
            self.results_dictionary()

        return self


# OQ LOGGER ###########################################


class UpdateMethod(ABC):
    @classmethod
    @abstractmethod
    def update(self):
        pass


class LoggerVariable(object):
    def __init__(
        self,
        attribute_name: str,
        history_update_method: UpdateMethod,
        best_update_method: UpdateMethod,
    ):
        self.name = attribute_name
        self.best = []
        self.history = []

        self.history_update_method = history_update_method
        self.best_update_method = best_update_method

    def update(self, new_value):
        self.update_history(new_value)
        self.update_best(new_value)

    def update_history(self, new_value):
        return self.history_update_method.update(self, "history", new_value)

    def update_best(self, new_value):
        return self.best_update_method.update(self, "best", new_value)


class AppendValue(UpdateMethod):
    def update(
        self, attribute_name: str, new_value: Union[list, int, float, str]
    ) -> None:
        get_var = getattr(self, attribute_name)
        get_var.append(new_value)
        setattr(self, attribute_name, get_var)


class ReplaceValue(UpdateMethod):
    def update(
        self, attribute_name: str, new_value: Union[list, int, float, str]
    ) -> None:
        setattr(self, attribute_name, [new_value])


class EmptyValue(UpdateMethod):
    def update(
        self, attribute_name: str, new_value: Union[list, int, float, str]
    ) -> None:
        setattr(self, attribute_name, [])


class IfLowerDo(UpdateMethod):
    def __init__(self, update_method: UpdateMethod):
        self._update_method = update_method

    def update(
        self,
        logger_variable: LoggerVariable,
        attribute_name: str,
        new_value: Union[int, float],
    ) -> None:
        try:
            old_value = getattr(logger_variable, attribute_name)[-1]
            if new_value < old_value:
                self._update_method.update(logger_variable, attribute_name, new_value)
        except IndexError:
            AppendValue.update(logger_variable, attribute_name, new_value)


class IfHigherDo(UpdateMethod):
    def __init__(self, update_method: UpdateMethod):
        self._update_method = update_method

    def update(
        self,
        logger_variable: LoggerVariable,
        attribute_name: str,
        new_value: Union[int, float],
    ) -> None:
        try:
            old_value = getattr(logger_variable, attribute_name)[-1]
            if new_value > old_value:
                self._update_method.update(logger_variable, attribute_name, new_value)
        except IndexError:
            AppendValue.update(logger_variable, attribute_name, new_value)


class LoggerVariableFactory(object):

    """
    Creates Logger Variable Objects
    """

    history_bool_mapping = {"True": AppendValue, "False": EmptyValue}
    best_string_mapping = {
        "HighestOnly": IfHigherDo(ReplaceValue),
        "HighestSoFar": IfHigherDo(AppendValue),
        "LowestOnly": IfLowerDo(ReplaceValue),
        "LowestSoFar": IfLowerDo(AppendValue),
        "Replace": ReplaceValue,
        "Append": AppendValue,
    }

    @classmethod
    def create_logger_variable(
        self, attribute_name: str, history_update_bool: bool, best_update_string: str
    ) -> LoggerVariable:
        history_update_method = self.history_bool_mapping[str(history_update_bool)]
        best_update_method = self.best_string_mapping[best_update_string]

        return LoggerVariable(attribute_name, history_update_method, best_update_method)


class Logger(object):
    def __init__(self, initialisation_variables: dict, logger_update_structure: dict):
        for attribute_name, attribute_properties in initialisation_variables.items():
            self._create_variable(
                attribute_name=attribute_name,
                history_update_bool=attribute_properties["history_update_bool"],
                best_update_string=attribute_properties["best_update_string"],
            )
        self._create_logger_update_structure(
            logger_update_structure["root_nodes"],
            logger_update_structure["best_update_structure"],
        )

    def _create_variable(
        self, attribute_name: str, history_update_bool: bool, best_update_string: str
    ):
        """
        If a variable was created was this method, ensure that the logger update
        strucuture is updated to include the new variable. Else the best value
        of the new variable will not be updated.
        """

        setattr(
            self,
            attribute_name,
            LoggerVariableFactory.create_logger_variable(
                attribute_name, history_update_bool, best_update_string
            ),
        )

    def _create_logger_update_structure(
        self, root_nodes: List[str], best_update_relations: Tuple[List[str]]
    ):
        """
        This method creates an internal representation based on the way in which
        the Logger Variables should be updated using a NetworkX Directed Graph.

        Parameter
        ---------
        root_nodes: `list`
            Starting points of the update graphs. The name of the Logger Variable
            that needs to be updated first should be named in this list.
        best_update_relations: `list`
            If the updating of B depends on A, (A changes, Update B). This
            relationship should be encapsuled/represented with the following
            list: ['A', 'B'].
        """

        self.best_update_structure = nx.DiGraph()

        self.root_nodes = root_nodes
        self.update_edges = best_update_relations

    @property
    def update_edges(self):
        return self._update_edges

    @update_edges.setter
    def update_edges(self, update_edges: Tuple[List[str]]):
        self.best_update_structure.add_edges_from(update_edges)
        self._update_edges = update_edges

        return self._update_edges

    @property
    def root_nodes(self):
        return self._root_nodes

    @root_nodes.setter
    def root_nodes(self, root_nodes: List[str]):
        self.best_update_structure.add_nodes_from(root_nodes)
        self._root_nodes = root_nodes

        return self._root_nodes

    def log_variables(self, input_dict: dict):
        """
        Updates all the histories of the logger variables.
        Best values of the logger variables are only updated according to the rules
        specified by its update structure. If the attribute is not in the update
        structure it isnt updated.

        input_dict:
            Should contain a dictionary with the key as the name of the
            attribute to be updated and its respective value. Note that the
            attribute should have been created beforehand.
        """

        for each_key in input_dict.keys():
            self._log_history(each_key, input_dict[each_key])

        # Updates based on best_update_structure, the networkx graph.
        change_dict = dict()

        node_list = deepcopy(self.root_nodes)

        while len(node_list) != 0:
            mid_list = []
            for each_node in node_list:
                change_bool = False

                # Checks if a nodes predecessor has been updated. If not all
                # of them were updated, the node is skipped.
                node_predecessor = self.best_update_structure.pred[each_node]
                updated_pred_count = np.sum(
                    [
                        change_dict[each_precessor_node]
                        for each_precessor_node in node_predecessor
                    ]
                )
                if len(node_predecessor) != updated_pred_count:
                    continue

                # Update node if attribute name is in input dictionary and
                # its predecessors hsa been updated.
                if each_node in input_dict.keys():
                    logged_var = getattr(self, each_node)
                    old_best = logged_var.best.copy()
                    self._log_best(each_node, input_dict[each_node])
                    new_best = logged_var.best
                    if type(new_best[0]) == np.ndarray:
                        if not np.array_equal(new_best, old_best):
                            change_bool = True
                    elif new_best != old_best:
                        change_bool = True

                # Keeps track of the nodes whose best values have changed
                change_dict[each_node] = change_bool

                # Retrieve the next set of nodes to update if the current node
                # was changed.
                if change_bool == True:
                    mid_list.extend(self.best_update_structure.adj[each_node].keys())

            node_list = list(set(mid_list))

    def _log_history(self, attribute_name: str, attribute_value):
        attr_to_update = getattr(self, attribute_name)
        attr_to_update.update_history(attribute_value)

    def _log_best(self, attribute_name: str, attribute_value):
        attr_to_update = getattr(self, attribute_name)
        attr_to_update.update_best(attribute_value)


# OQ DERIVATIVE FUNCTIONS ###########################


def update_and_compute_expectation(
    backend_obj: QAOABaseBackend, params: QAOAVariationalBaseParams, logger: Logger
):
    """
    Helper function that returns a callable that takes in a list/nparray of raw parameters.
    This function will handle:

        #. Updating logger object with `logger.log_variables`
        #. Updating variational parameters with `update_from_raw`
        #. Computing expectation with `backend_obj.expectation`

    Parameters
    ----------
    backend_obj: QAOABaseBackend
        `QAOABaseBackend` object that contains information about the
        backend that is being used to perform the QAOA circuit
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
    logger: Logger
        Logger Class required to log information from the evaluations
        required for the jacobian/hessian computation.
    Returns
    -------
    out:
        A callable that accepts a list/array of parameters, and returns the
        computed expectation value.
    """

    def fun(args, n_shots=None):
        current_total_eval = logger.func_evals.best[0]
        current_total_eval += 1
        current_jac_eval = logger.jac_func_evals.best[0]
        current_jac_eval += 1
        logger.log_variables(
            {"func_evals": current_total_eval, "jac_func_evals": current_jac_eval}
        )
        params.update_from_raw(args)

        n_shots_dict = {"n_shots": n_shots} if n_shots else {}
        return backend_obj.expectation(params, **n_shots_dict)

    return fun


def update_and_get_counts(
    backend_obj: QAOABaseBackend, params: QAOAVariationalBaseParams, logger: Logger
):
    """
    Helper function that returns a callable that takes in a list/nparray of
    raw parameters.
    This function will handle:

        #. Updating logger object with `logger.log_variables`
        #. Updating variational parameters with `update_from_raw`
        #. Getting the counts dictonary with `backend_obj.get_counts`

    PARAMETERS
    ----------
    backend_obj: QAOABaseBackend
        `QAOABaseBackend` object that contains information about the backend
        that is being used to perform the QAOA circuit
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
    logger: Logger
        Logger Class required to log information from the evaluations
        required for the jacobian/hessian computation.

    Returns
    -------
    out:
        A callable that accepts a list/array of parameters,
        and returns the counts dictonary.
    """

    def fun(args, n_shots=None):
        current_total_eval = logger.func_evals.best[0]
        current_total_eval += 1
        current_jac_eval = logger.jac_func_evals.best[0]
        current_jac_eval += 1
        logger.log_variables(
            {"func_evals": current_total_eval, "jac_func_evals": current_jac_eval}
        )
        params.update_from_raw(args)

        n_shots_dict = {"n_shots": n_shots} if n_shots else {}
        return backend_obj.get_counts(params, **n_shots_dict)

    return fun


def derivative(
    backend_obj: QAOABaseBackend,
    params: QAOAVariationalBaseParams,
    logger: Logger,
    derivative_type: str = None,
    derivative_method: str = None,
    derivative_options: dict = None,
):
    """
    Returns a callable function that calculates the gradient according
    to the specified `gradient_method`.

    Parameters
    ----------
    backend_obj: QAOABaseBackend
        `QAOABaseBackend` object that contains information about the
        backend that is being used to perform the QAOA circuit
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
    logger: Logger
        Logger Class required to log information from the evaluations
        required for the jacobian/hessian computation.
    derivative_type : str
        Type of derivative to compute. Either `gradient` or `hessian`.
    derivative_method : str
        Computational method of the derivative.
        Either `finite_difference`, `param_shift`, `stoch_param_shift`, or `grad_spsa`.
    derivative_options : dict
        Dictionary containing options specific to each `derivative_method`.
    cost_std :
        object that computes expectation values when executed. Standard parametrisation.
    cost_ext :
        object that computes expectation values when executed. Extended parametrisation.
        Mainly used to compute parameter shifts at each individual gate,
        which is summed to recover the parameter shift for a parametrised layer.
    Returns
    -------
    out:
        The callable derivative function of the cost function,
        generated based on the `derivative_type`, `derivative_method`,
        and `derivative_options` specified.
    """
    # Default derivative_options used if none are specified.
    default_derivative_options = {
        "stepsize": 0.00001,
        "n_beta_single": -1,
        "n_beta_pair": -1,
        "n_gamma_single": -1,
        "n_gamma_pair": -1,
    }

    derivative_options = (
        {**default_derivative_options, **derivative_options}
        if derivative_options is not None
        else default_derivative_options
    )

    # cost_std = derivative_dict['cost_std']
    # cost_ext = derivative_dict['cost_ext']
    params_ext = QAOAVariationalExtendedParams.empty(backend_obj.qaoa_descriptor)

    derivative_types = ["gradient", "gradient_w_variance", "hessian"]
    assert (
        derivative_type in derivative_types
    ), "Unknown derivative type specified - please choose between " + str(
        derivative_types
    )

    derivative_methods = [
        "finite_difference",
        "param_shift",
        "stoch_param_shift",
        "grad_spsa",
    ]
    assert derivative_method in derivative_methods, (
        "Unknown derivative computation method specified - please choose between "
        + str(derivative_methods)
    )

    params = deepcopy(params)

    if derivative_type == "gradient":
        if derivative_method == "finite_difference":
            out = grad_fd(backend_obj, params, derivative_options, logger)
        elif derivative_method == "param_shift":
            assert (
                params.__class__.__name__ == "QAOAVariationalStandardParams"
            ), f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_ps(backend_obj, params, params_ext, derivative_options, logger)
        elif derivative_method == "stoch_param_shift":
            assert (
                params.__class__.__name__ == "QAOAVariationalStandardParams"
            ), f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_ps(
                backend_obj,
                params,
                params_ext,
                derivative_options,
                logger,
                stochastic=True,
            )
        elif derivative_method == "grad_spsa":
            out = grad_spsa(backend_obj, params, derivative_options, logger)

    elif derivative_type == "gradient_w_variance":
        if derivative_method == "finite_difference":
            out = grad_fd(
                backend_obj, params, derivative_options, logger, variance=True
            )
        elif derivative_method == "param_shift":
            assert (
                params.__class__.__name__ == "QAOAVariationalStandardParams"
            ), f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_ps(
                backend_obj,
                params,
                params_ext,
                derivative_options,
                logger,
                variance=True,
            )
        elif derivative_method == "stoch_param_shift":
            assert (
                params.__class__.__name__ == "QAOAVariationalStandardParams"
            ), f"{params.__class__.__name__} not supported - only Standard Parametrisation is supported for parameter shift/stochastic parameter shift for now."
            out = grad_ps(
                backend_obj,
                params,
                params_ext,
                derivative_options,
                logger,
                stochastic=True,
                variance=True,
            )
        elif derivative_method == "grad_spsa":
            out = grad_spsa(
                backend_obj, params, derivative_options, logger, variance=True
            )

    elif derivative_type == "hessian":
        if derivative_method == "finite_difference":
            out = hessian_fd(backend_obj, params, derivative_options, logger)
        else:
            raise ValueError(
                "Only support hessian derivative method is finite_difference. Your choice: {}".format(
                    derivative_method
                )
            )

    return out


def __create_n_shots_list(n_params, n_shots):
    """
    Creates a list of `n_shots` for each parameter in the variational circuit.

    Parameters
    ----------
    n_params : int
        Number of parameters in the variational circuit.
    n_shots : int or list
        Number of shots to use for each parameter in the variational circuit.
        If `n_shots` is an integer, then the same number of shots is used for each parameter.
        If `n_shots` is a list, then the length of the list must be equal to `n_params`
        and we return the list.

    Returns
    -------
    out:
        List of `n_shots` for each parameter in the variational circuit.
    """

    # If n_shots is a list, then it must be of length n_params,
    # else create a list of length n_params with all elements equal to n_shots
    if isinstance(n_shots, list):
        assert (
            len(n_shots) == n_params
        ), "n_shots must be a list of length equal to the number of parameters."
        n_shots_list = n_shots
    elif isinstance(n_shots, int) or n_shots is None:
        n_shots_list = [n_shots] * n_params
    else:
        raise ValueError("n_shots must be either an integer or a list of integers.")

    return n_shots_list


def __create_n_shots_ext_list(n_params, n_associated_params, n_shots):
    """
    Creates a list of number of shots for each parameter in the extended parametrisation.
    If n_shots is a integer, then it is used for all extended parameters. So, we create a list of length sum(n_associated_params)
    with all elements equal to n_shots. (sum(n_associated_params) = number of extended params)
    If n_shots is a list, then this list tell us the number of shots for each standard parameter. We convert this list to a list of
    number of shots for each extended parameter. Each standard parameter has a different number of associated extended parameters,
    `n_associated_params` helps us with this. Each element of `n_associated_params` is the number of associated extended parameters to each coefficient.
    And we know that each standard parameter has 2 associated coefficients (mixer_1q, mixer_2q, cost_1q, cost_2q).

    Parameters
    ----------
    n_associated_params : list
        List of integers, where each integer is the number of associated parameters in the extended parametrisation for each coefficient.
        The sum of all elements in this list is equal to the number of parameters in the extended parametrisation.
    n_params : int
        Number of parameters in the standard parametrisation.
    n_shots : int or list
        Number of shots to use for each parameter in the standard parametrisation.
        If an integer, then it is used for all the extended parameters.
        If a list, then it must be of length sum(n_associated_params).

    Returns
    -------
    n_shots_ext_list : list
        List of integers, where each integer is the number of shots to use for each parameter in the extended parametrisation.
    """

    # If n_shots is a list, then it must be of length n_params, else create a list of length n_params with all elements equal to n_shots
    if isinstance(n_shots, list):
        assert (
            len(n_shots) == n_params
        ), "n_shots must be a list of length equal to the number of parameters."

        # transform n_shots list (which has length n_params) into a list of the same length of n_associated_params
        n_shots = np.array(n_shots)
        n_shots = (
            n_shots.reshape(2, len(n_shots) // 2)
            .repeat(2, axis=0)
            .reshape(n_shots.size * 2)
        )  # repeat each element twice in the following way: if we have [1,2,3,4] we get [1,2,1,2,3,4,3,4]

        # create the list of n_shots for each parameter in the extended parametrisation.
        # For each parameter of each coefficient we have a number of shots (`n_shots` list),
        # we need to repeat this number of shots the number of associated extended parameters
        # that each coefficient has (`n_associated_params` list).
        n_shots_list = [
            shots for r, shots in zip(n_associated_params, n_shots) for _ in range(r)
        ]
    elif isinstance(n_shots, int) or n_shots is None:
        n_shots_list = [n_shots] * np.sum(n_associated_params)
    else:
        raise ValueError("n_shots must be either an integer or a list of integers.")

    return n_shots_list


def __gradient(args, backend_obj, params, logger, variance):
    """
    Returns a callable function that computes the gradient as `constant*(fun(args + vect_eta) - fun(args - vect_eta))`
    and its variance (if variance=True).
    Where fun is the function that we want to differentiate,
    vect_eta is a vector of length n_params, and constant is a
    constant that depends on the derivative method.

    Parameters
    ----------
    args : np.array
        List of arguments to pass to the function that we want to differentiate.
    backend_obj : QAOABaseBackend
        `QAOABaseBackend` object that contains information about the backend that
        is being used to perform the QAOA circuit
    params : QAOAVariationalBaseParams
        `QAOAVariationalBaseParams` object containing variational angles.
    logger: Logger
        Logger Class required to log information from the evaluations
        required for the jacobian/hessian computation.
    variance : bool
        If True, then the variance of the gradient is also computed.

    Returns
    -------
    out:
        Callable function that computes the gradient and its variance (if variance=True).
    """

    def fun_w_variance(vect_eta, constant, n_shots=None):
        """
        Computes the gradient and its variance as `constant*(fun(args + vect_eta) - fun(args - vect_eta))`.

        Parameters
        ----------
        vect_eta : np.array
            Vector of length n_params.
        constant : float
            Constant that depends on the derivative method.
        n_shots : int
            Number of shots to use when calling fun().

        Returns
        -------
        out:
            Gradient and its variance.
        """

        # get value of eta, the function to get counts, hamiltonian, and alpha
        fun = update_and_get_counts(backend_obj, params, logger)
        hamiltonian = backend_obj.qaoa_descriptor.cost_hamiltonian
        alpha = backend_obj.cvar_alpha

        # get counts f(x+eta/2) and f(x-eta/2)
        counts_i_dict = fun(args - vect_eta, n_shots=n_shots)
        counts_f_dict = fun(args + vect_eta, n_shots=n_shots)

        # compute cost for each state in the counts dictionaries
        costs_dict = {
            key: cost_function({key: 1}, hamiltonian, alpha)
            for key in counts_i_dict.keys() | counts_f_dict.keys()
        }

        # for each count get the cost and create a list of shot costs
        eval_i_list = [
            costs_dict[key]
            for key, value in counts_i_dict.items()
            for _ in range(value)
        ]
        eval_f_list = [
            costs_dict[key]
            for key, value in counts_f_dict.items()
            for _ in range(value)
        ]

        # check if the number of shots used in the simulator / QPU is equal to n_shots
        assert (
            len(eval_i_list) == n_shots and len(eval_f_list) == n_shots
        ), "This backend does not support changing the number of shots."

        # compute a list of gradients of one shot cost
        grad_list = np.real(constant * (np.array(eval_f_list) - np.array(eval_i_list)))

        # return average and variance for the gradient for this argument
        return np.mean(grad_list), np.var(grad_list)

    def fun(vect_eta, constant, n_shots=None):
        """
        Computes the gradient as `constant*(fun(args + vect_eta) - fun(args - vect_eta))`.

        Parameters
        ----------
        vect_eta : np.array
            Vector of length n_params.
        constant : float
            Constant that depends on the derivative method.
        n_shots : int
            Number of shots to use when calling fun().

        Returns
        -------
        out:
            Gradient.
        """
        fun = update_and_compute_expectation(backend_obj, params, logger)
        return (
            constant
            * (
                fun(args + vect_eta, n_shots=n_shots)
                - fun(args - vect_eta, n_shots=n_shots)
            ),
            0,
        )

    if variance:
        return fun_w_variance
    else:
        return fun


def grad_fd(backend_obj, params, gradient_options, logger, variance: bool = False):
    """
    Returns a callable function that calculates the gradient (and its variance if `variance=True`)
    with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed.
    params : `QAOAVariationalBaseParams`
        parameters of the variational circuit.
    gradient_options : `dict`
        stepsize :
            Stepsize of finite difference.
    logger : `Logger`
        logger object to log the number of function evaluations.
    variance : `bool`
        If True, the variance of the gradient is also computed.
        If False, only the gradient is computed.

    RETURNS
    -------
    grad_fd_func: `Callable`
        Callable derivative function.
    """

    # Set default value of eta
    eta = gradient_options["stepsize"]

    def grad_fd_func(args, n_shots=None):
        # get the function to compute the gradient and its variance
        __gradient_function = __gradient(args, backend_obj, params, logger, variance)

        # if n_shots is int or None create a list with len of args
        # (if it is none, it will use the default n_shots)
        n_shots_list = __create_n_shots_list(len(args), n_shots)

        # lists of gradients and variances for each argument, initialized with zeros
        grad, var = np.zeros(len(args)), np.zeros(len(args))

        for i in range(len(args)):
            # vector and constant to compute the gradient for the i-th argument
            vect_eta = np.zeros(len(args))
            vect_eta[i] = eta / 2
            const = 1 / eta

            # Finite diff. calculation of gradient
            grad[i], var[i] = __gradient_function(
                vect_eta, const, n_shots_list[i]
            )  # const*[f(args + vect_eta) - f(args - vect_eta)]

        # if variance is True, add the number of shots per argument to the logger
        if variance:
            logger.log_variables({"n_shots": n_shots_list})

        if variance:
            return (
                grad,
                var,
                2 * sum(n_shots_list),
            )  # return gradient, variance, and total number of shots
        else:
            return grad  # return gradient

    return grad_fd_func


def grad_ps(
    backend_obj,
    params,
    params_ext,
    gradient_options,
    logger,
    stochastic: bool = False,
    variance: bool = False,
):
    """
    If `stochastic=False` returns a callable function that calculates the gradient
    (and its variance if `variance=True`) with the parameter shift method.
    If `stochastic=True` returns a callable function that approximates the gradient
    (and its variance if `variance=True`) with the stochastic parameter shift method,
    which samples (n_beta_single, n_beta_pair, n_gamma_single, n_gamma_pair) gates at
    each layer instead of all gates. See "Algorithm 4" of https://arxiv.org/pdf/1910.01155.pdf.
    By convention, (n_beta_single, n_beta_pair, n_gamma_single, n_gamma_pair) = (-1, -1, -1, -1)
    will sample all gates (which is then equivalent to the full parameter shift rule).

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed.
    params : `QAOAVariationalStandardParams`
        variational parameters object, standard parametrisation.
    params_ext : `QAOAVariationalExtendedParams`
        variational parameters object, extended parametrisation.
    gradient_options :
        * 'n_beta_single': Number of single-qubit mixer gates to sample for the stochastic parameter shift.
        * 'n_beta_pair': Number of two-qubit mixer gates to sample for the stochastic parameter shift.
        * 'n_gamma_single': Number of single-qubit cost gates to sample for the stochastic parameter shift.
        * 'n_gamma_pair': Number of two-qubit cost gates to sample for the stochastic parameter shift.
    logger : `Logger`
        logger object to log the number of function evaluations.
    variance : `bool`
        If True, the variance of the gradient is also computed.
        If False, only the gradient is computed.

    RETURNS
    -------
    grad_ps_func:
        Callable derivative function.
    """
    # TODO : handle Fourier parametrisation

    # list of list of the coefficients (mixer_1q, mixer_2q, cost_1q, cost_2q)
    params_coeffs = [
        params.mixer_1q_coeffs,
        params.mixer_2q_coeffs,
        params.cost_1q_coeffs,
        params.cost_2q_coeffs,
    ]
    # list of coefficients for all the extended parameters
    coeffs_list = [
        x for coeffs in params_coeffs for x in coeffs for _ in range(params.p)
    ]

    # create a list of how many extended parameters are associated with each coefficient
    n_associated_params = np.repeat([len(x) for x in params_coeffs], params.p)
    # with the `n_associated_params` list, add a 0 in the first position and
    # sum the list cumulatively (so that this list indicate which gate is associated with which coefficient)
    l = np.insert(
        np.cumsum(n_associated_params), 0, 0
    )  # the i-th coefficient is associated with extended parameters with indices in range [l[i], l[i+1]]

    def grad_ps_func(args, n_shots=None):
        # Convert standard to extended parameters before applying parameter shift
        args_ext = params.convert_to_ext(args)

        # get the function to compute the gradient and its variance
        __gradient_function = __gradient(
            args_ext, backend_obj, params_ext, logger, variance
        )

        # we call the function that returns the number of shots for each extended parameter,
        # giving the number of shots for each standard parameter (n_shots)
        n_shots_list = __create_n_shots_ext_list(
            len(args), n_associated_params, n_shots
        )

        # compute the gradient (and variance) of all extended parameters
        # with the stochastic parameter shift method lists of gradients and variances
        # for each argument (extended), initialized with zeros
        grad_ext, var_ext = np.zeros(len(args_ext)), np.zeros(len(args_ext))
        # variable to count the number of shots used
        n_shots_used = 0
        # Apply parameter shifts
        for i, sample_param_bool in enumerate(__sample_params()):
            # __sample_params is only relevant if stochastic=True, if stochastic=False
            # we loop over all extended parameters. __sample_params() returns a list with True or False
            # for each extended parameter (`i` is the index of the extended parameter,
            # and `sample_param_bool` is True if the parameter is sampled)
            if not sample_param_bool:
                # (only relevant if stochastic=True) if the parameter is not
                # sampled (`sample_param_bool==False`), skip it
                continue
            r = coeffs_list[i]
            vect_eta = np.zeros(len(args_ext))
            vect_eta[i] = np.pi / (4 * r)
            grad_ext[i], var_ext[i] = __gradient_function(
                vect_eta, r, n_shots_list[i]
            )  # r*[f(args + vect_eta) - f(args - vect_eta)]
            n_shots_used += (
                n_shots_list[i] * 2 if n_shots_list[i] != None else 0
            )  # we multiply by 2 because we call the function twice to compute the gradient

        ## convert extended form back into standard form
        # sum all the gradients for each parameter of each coefficient according to the indices in l,
        # and rehape the array to 4 rows and p columns, first row is mixer_1q, second is mixer_2q,
        # third is cost_1q, fourth is cost_2q
        grad = np.array(
            [np.sum(grad_ext[l[i] : l[i + 1]]) for i in range(len(l) - 1)]
        ).reshape(4, params.p)
        # summing 1q with 2q gradients (first row with second row, third row with fourth row),
        # to get the gradient for each parameter in the standard form
        grad = np.concatenate(
            (grad[0] + grad[1], grad[2] + grad[3])
        )  # now we have the gradient in the standard form (one gradient for each standard parameter)
        # repeat the same for the variances
        var = np.array(
            [np.sum(var_ext[l[i] : l[i + 1]]) for i in range(len(l) - 1)]
        ).reshape(4, params.p)
        var = np.concatenate((var[0] + var[1], var[2] + var[3]))

        # if variance is True, add the number of shots per argument (standard) to the logger
        if variance:
            logger.log_variables({"n_shots": __create_n_shots_list(len(args), n_shots)})

        if variance:
            return (
                grad,
                var,
                n_shots_used,
            )  # return gradient, variance, and total number of shots
        else:
            return grad  # return gradient

    def __sample_params():
        """
        Generator that returns a list with True or False for each extended parameter,
        if False the parameter is not sampled, meaning that the gradient is not computed for that parameter.
        For parameter shift method (not stochastic), all parameters are sampled.
        For stochastic parameter shift method, the parameters are sampled according
        to the number given by the user for each parameter.
        """

        if stochastic:
            # if stochastic is True, we sample the parameters according to the number given by the
            # user for each parameter, this information is in the `gradient_options` dictionary.
            # list of the names of the parameters and list of the number of gates for each parameter
            names_params = [
                "n_beta_single",
                "n_beta_pair",
                "n_gamma_single",
                "n_gamma_pair",
            ]
            n_sample = [gradient_options[x] for x in names_params]

            # check if the number of gates to sample is valid and if it is -1
            # then set it to the number of gates in the full parameter shift rule
            for i, x in enumerate(n_sample):
                y = len(params_coeffs[i])
                assert (
                    -1 <= x <= y
                ), f"Invalid jac_options['{names_params[i]}'], it must be between -1 and {y}, but {x} is passed."
                if x == -1:
                    n_sample[i] = y

            # create a list of how many gates are associated with each coefficient
            n_sample = np.repeat(n_sample, params.p)

            # sample the parameters according to the number of gates to sample (this information is in n_sample)
            sampled_list = np.array([])
            for i, n in enumerate(n_sample):
                # randomly choose n parameters to sample in the i-th group of standard parameters
                sampled_indices = np.random.choice(range(l[i], l[i + 1]), n, False)
                # create a list of booleans, where True means that the parameter is sampled
                sampled_list = np.append(
                    sampled_list,
                    np.array([i in sampled_indices for i in range(l[i], l[i + 1])]),
                )
        else:
            # if stochastic is False, sample all the parameters (there is one parameter for each coefficient)
            sampled_list = np.ones(len(coeffs_list), dtype=bool)

        return sampled_list

    return grad_ps_func


def grad_spsa(backend_obj, params, gradient_options, logger, variance: bool = False):
    """
    Returns a callable function that calculates the gradient approxmiation with the
    Simultaneous Perturbation Stochastic Approximation (SPSA) method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed.
    params : `QAOAVariationalBaseParams`
        variational parameters object.
    gradient_options : `dict`
        gradient_stepsize :
            stepsize of stochastic shift.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    grad_spsa_func: `Callable`
        Callable derivative function.

    """
    eta = gradient_options["stepsize"]

    def grad_spsa_func(args, n_shots=None):
        # if variance is True, add the number of shots per argument to the logger
        if variance:
            logger.log_variables({"n_shots": [n_shots]})

        # get the function to compute the gradient and its variance
        __gradient_function = __gradient(args, backend_obj, params, logger, variance)

        # vector and constant to compute the gradient and its variance
        delta = 2 * np.random.randint(0, 2, size=len(args)) - 1
        vector_eta = delta * eta / 2
        const = 1 / eta

        # compute the gradient and its variance: const*[f(args + vect_eta) - f(args - vect_eta)]
        grad, var = __gradient_function(vector_eta, const, n_shots)

        if variance:
            return (
                grad * delta,
                var * np.abs(delta),
                2 * n_shots,
            )  # return the gradient, its variance and the total number of shots
        else:
            return grad * delta  # return the gradient

    return grad_spsa_func


def hessian_fd(backend_obj, params, hessian_options, logger):
    """
    Returns a callable function that calculates the hessian with the finite difference method.

    PARAMETERS
    ----------
    backend_obj : `QAOABaseBackend`
        backend object that computes expectation values when executed.
    params : `QAOAVariationalBaseParams`
        variational parameters object.
    hessian_options :
        hessian_stepsize :
            stepsize of finite difference.
    logger : `Logger`
        logger object to log the number of function evaluations.

    RETURNS
    -------
    hessian_fd_func:
        Callable derivative function.

    """

    eta = hessian_options["stepsize"]
    fun = update_and_compute_expectation(backend_obj, params, logger)

    def hessian_fd_func(args):
        hess = np.zeros((len(args), len(args)))

        for i in range(len(args)):
            for j in range(len(args)):
                vect_eta1 = np.zeros(len(args))
                vect_eta2 = np.zeros(len(args))
                vect_eta1[i] = 1
                vect_eta2[j] = 1

                if i == j:
                    # Central diff. hessian diagonals (https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm)
                    hess[i][i] = (
                        -fun(args + 2 * eta * vect_eta1)
                        + 16 * fun(args + eta * vect_eta1)
                        - 30 * fun(args)
                        + 16 * fun(args - eta * vect_eta1)
                        - fun(args - 2 * eta * vect_eta1)
                    ) / (12 * eta**2)
                # grad_diff[i] = (grad_fd_ext(params + (eta/2)*vect_eta)[i] - grad_fd_ext(params - (eta/2)*vect_eta)[i])/eta
                else:
                    hess[i][j] = (
                        fun(args + eta * vect_eta1 + eta * vect_eta2)
                        - fun(args + eta * vect_eta1 - eta * vect_eta2)
                        - fun(args - eta * vect_eta1 + eta * vect_eta2)
                        + fun(args - eta * vect_eta1 - eta * vect_eta2)
                    ) / (4 * eta**2)

        return hess

    return hessian_fd_func


# OQ QFIM #######################################


def log_qfim_evals(logger: Logger) -> Logger:
    current_total_eval = logger.func_evals.best[0]
    current_total_eval += 1
    current_qfim_eval = logger.qfim_func_evals.best[0]
    current_qfim_eval += 1
    logger.log_variables(
        {"func_evals": current_total_eval, "qfim_func_evals": current_qfim_eval}
    )

    return logger


def qfim(
    backend_obj: QAOABaseBackend,
    params: QAOAVariationalBaseParams,
    logger: Logger,
    eta: float = 0.00000001,
):
    """
    Returns a callable qfim_fun(args) that computes the
    Quantum Fisher Information Matrix at `args` according to :
    $$[QFI]_{ij} = Re(<∂iφ|∂jφ>) − <∂iφ|φ><φ|∂jφ>$$.

    Parameters
    ----------
    params: `QAOAVariationalBaseParams`
        The QAOA parameters as a 1D array (derived from an object of one of
        the parameter classes, containing hyperparameters and variable parameters).

    eta: `float`
        The infinitesimal shift used to compute `|∂jφ>`, the partial derivative
        of the wavefunction w.r.t a parameter.

    Returns
    -------
    qfim_array:
        The quantum fisher information matrix, a 2p*2p symmetric square matrix
        with elements [QFI]_ij = Re(<∂iφ|∂jφ>) − <∂iφ|φ><φ|∂jφ>.
    """

    if isinstance(backend_obj, QAOABaseBackendShotBased):
        raise NotImplementedError(
            "QFIM computation is not currently available on shot-based"
        )

    psi = backend_obj.wavefunction(params)
    log_qfim_evals(logger)

    qfim_array = np.zeros((len(params), len(params)))

    copied_params = deepcopy(params)

    def qfim_fun(args):
        for i in range(len(args)):
            for j in range(i + 1):
                vi, vj = np.zeros(len(args)), np.zeros(len(args))
                vi[i] = eta
                vj[j] = eta

                copied_params.update_from_raw(args + vi)
                wavefunction_plus_i = np.array(backend_obj.wavefunction(copied_params))
                log_qfim_evals(logger)

                copied_params.update_from_raw(args - vi)
                wavefunction_minus_i = np.array(backend_obj.wavefunction(copied_params))
                log_qfim_evals(logger)

                copied_params.update_from_raw(args + vj)
                wavefunction_plus_j = np.array(backend_obj.wavefunction(copied_params))
                log_qfim_evals(logger)

                copied_params.update_from_raw(args - vj)
                wavefunction_minus_j = np.array(backend_obj.wavefunction(copied_params))
                log_qfim_evals(logger)

                di_psi = (wavefunction_plus_i - wavefunction_minus_i) / eta
                dj_psi = (wavefunction_plus_j - wavefunction_minus_j) / eta

                qfim_array[i][j] = np.real(np.vdot(di_psi, dj_psi)) - np.vdot(
                    di_psi, psi
                ) * np.vdot(psi, dj_psi)

                if i != j:
                    qfim_array[j][i] = qfim_array[i][j]

        return qfim_array

    return qfim_fun


# OQ RESULT ####################################


def most_probable_bitstring(cost_hamiltonian, measurement_outcomes):
    """
    Computing the most probable bitstring
    """
    mea_out = list(measurement_outcomes.values())
    index_likliest_states = np.argwhere(mea_out == np.max(mea_out))
    # degeneracy = len(index_likliest_states)
    solutions_bitstrings = [
        list(measurement_outcomes.keys())[e[0]] for e in index_likliest_states
    ]

    return {
        "solutions_bitstrings": solutions_bitstrings,
        "bitstring_energy": bitstring_energy(cost_hamiltonian, solutions_bitstrings[0]),
    }


# GET OPTIMIZER ############################################


def available_optimizers():
    """
    Return a list of available optimizers.
    """

    optimizers = {
        "scipy": ScipyOptimizer.SCIPY_METHODS,
    }

    return optimizers


def get_optimizer(
    vqa_object: VQABaseBackend,
    variational_params: QAOAVariationalBaseParams,
    optimizer_dict: dict,
    publisher: Publisher,
):
    """
    Initialise the specified optimizer class with provided method and
    optimizer-specific options

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the
        backend used to perform computations, and the VQA circuit.

    variational_params:
        Object of class QAOAVariationalBaseParams, which contains
        information on the circuit to be executed,
        the type of parametrisation, and the angles of the VQA circuit.

    optimizer_dict:
        Optimizer information dictionary used to construct
        the optimizer with specified options

    Returns
    -------
    optimizer:
        Optimizer object of type specified by specified method
    """
    SUPPORTED_OPTIMIZERS = {
        "scipy": ScipyOptimizer,
    }

    method = optimizer_dict["method"].lower()
    optimizers = available_optimizers()

    method_valid = False
    for opt_class, methods in optimizers.items():
        if method in methods:
            selected_class = opt_class
            method_valid = True

    assert method_valid, ValueError(
        f"Selected optimizer method '{method}' is not supported."
        f"Please choose from {available_optimizers()}"
    )

    optimizer = SUPPORTED_OPTIMIZERS[selected_class](
        vqa_object, variational_params, optimizer_dict,
        publisher
    )

    return optimizer


class QAOAResult:
    """
    A class to handle the results of QAOA workflows

    Parameters
    ----------
    log: `Logger`
        The raw logger generated from the training vqa part of the QAOA.
    method: `str`
        Stores the name of the optimisation used by the classical optimiser
    cost_hamiltonian: `Hamiltonian`
        The cost Hamiltonian for the problem statement
    type_backend: `QAOABaseBackend`
        The type of backend used for the experiment
    """

    def __init__(
        self,
        log: "Logger",
        method: Type[str],
        cost_hamiltonian: Type[Hamiltonian],
        type_backend: Type[QAOABaseBackend],
    ):
        """
        init method
        """
        self.__type_backend = type_backend
        self.method = method
        self.cost_hamiltonian = cost_hamiltonian

        self.evals = {
            "number_of_evals": log.func_evals.best[0],
            "jac_evals": log.jac_func_evals.best[0],
            "qfim_evals": log.qfim_func_evals.best[0],
        }

        self.intermediate = {
            "angles": np.array(log.param_log.history).tolist(),
            "cost": log.cost.history,
            "measurement_outcomes": log.measurement_outcomes.history,
            "job_id": log.job_ids.history,
        }

        self.optimized = {
            "angles": np.array(log.param_log.best[0]).tolist()
            if log.param_log.best != []
            else [],
            "cost": log.cost.best[0] if log.cost.best != [] else None,
            "measurement_outcomes": log.measurement_outcomes.best[0]
            if log.measurement_outcomes.best != []
            else {},
            "job_id": log.job_ids.best[0] if len(log.job_ids.best) != 0 else [],
            "eval_number": log.eval_number.best[0]
            if len(log.eval_number.best) != 0
            else [],
        }

        self.most_probable_states = (
            most_probable_bitstring(
                cost_hamiltonian, self.get_counts(log.measurement_outcomes.best[0])
            )
            if log.measurement_outcomes.best != []
            else []
        )
        # def __repr__(self):
        #     """Return an overview over the parameters and hyperparameters
        #     Todo
        #     ----
        #     Split this into ``__repr__`` and ``__str__`` with a more verbose
        #     output in ``__repr__``.
        #     """
        #     string = "Optimization Results:\n"
        #     string += "\tThe solution is " + str(self.solution['degeneracy']) + " degenerate" "\n"
        #     string += "\tThe most probable bitstrings are: " + str(self.solution['bitstring']) + "\n"
        #     string += "\tThe associated cost is: " + str(self.optimized['cost']) + "\n"

        #     return (string)

        # if we are using a shot adaptive optimizer, we need to add the number of shots to the result
        if log.n_shots.history != []:
            self.n_shots = log.n_shots.history

    def asdict(
        self,
        keep_cost_hamiltonian: bool = True,
        complex_to_string: bool = False,
        intermediate_measurements: bool = True,
        exclude_keys: List[str] = [],
    ):
        """
        Returns a dictionary with the results of the optimization, where the dictionary is serializable.
        If the backend is a statevector backend, the measurement outcomes will be the statevector,
        meaning that it is a list of complex numbers, which is not serializable.
        If that is the case, and complex_to_string is true the complex numbers are converted to strings.

        Parameters
        ----------
        keep_cost_hamiltonian: `bool`
            If True, the cost hamiltonian is kept in the dictionary. If False, it is removed.
        complex_to_string: `bool`
            If True, the complex numbers are converted to strings. If False, they are kept as complex numbers.
            This is useful for the JSON serialization.
        intermediate_measurements: bool, optional
            If True, intermediate measurements are included in the dump.
            If False, intermediate measurements are not included in the dump.
            Default is True.
        exclude_keys: `list[str]`
            A list of keys to exclude from the returned dictionary.

        Returns
        -------
        return_dict: `dict`
            A dictionary with the results of the optimization, where the dictionary is serializable.
        """

        return_dict = {}
        return_dict["method"] = self.method
        if keep_cost_hamiltonian:
            return_dict["cost_hamiltonian"] = convert2serialize(self.cost_hamiltonian)
        return_dict["evals"] = self.evals
        return_dict["most_probable_states"] = self.most_probable_states

        complex_to_str = (
            lambda x: str(x)
            if isinstance(x, np.complex128) or isinstance(x, complex)
            else x
        )

        # if the backend is a statevector backend, the measurement outcomes will be the statevector,
        # meaning that it is a list of complex numbers, which is not serializable.
        # If that is the case, and complex_to_string is true the complex numbers are converted to strings.
        if complex_to_string:
            return_dict["intermediate"] = {}
            for key, value in self.intermediate.items():
                # Measurements and Cost may require casting
                if "measurement" in key:
                    if len(value) > 0:
                        if intermediate_measurements is False:
                            # if intermediate_measurements is false, the intermediate measurements are not included
                            return_dict["intermediate"][key] = []
                        elif isinstance(
                            value[0], np.ndarray
                        ):  # Statevector -> convert complex to str
                            return_dict["intermediate"][key] = [
                                [complex_to_str(item) for item in list_]
                                for list_ in value
                                if (
                                    isinstance(list_, list)
                                    or isinstance(list_, np.ndarray)
                                )
                            ]
                        else:  # All other case -> cast numpy into
                            return_dict["intermediate"][key] = [
                                {k_: int(v_) for k_, v_ in v.items()} for v in value
                            ]
                    else:
                        pass
                elif "cost" == key and (
                    isinstance(value[0], np.float64) or isinstance(value[0], np.float32)
                ):
                    return_dict["intermediate"][key] = [float(item) for item in value]
                else:
                    return_dict["intermediate"][key] = value

            return_dict["optimized"] = {}
            for key, value in self.optimized.items():
                # If wavefunction do complex to str
                if "measurement" in key and (
                    isinstance(value, list) or isinstance(value, np.ndarray)
                ):
                    return_dict["optimized"][key] = [
                        complex_to_str(item) for item in value
                    ]
                # if dictionary, convert measurement values to integers
                elif "measurement" in key and (isinstance(value, dict)):
                    return_dict["optimized"][key] = {
                        k: int(v) for k, v in value.items()
                    }
                else:
                    return_dict["optimized"][key] = value

                if "cost" in key and (
                    isinstance(value, np.float64) or isinstance(value, np.float32)
                ):
                    return_dict["optimized"][key] = float(value)
        else:
            return_dict["intermediate"] = self.intermediate
            return_dict["optimized"] = self.optimized

        # if we are using a shot adaptive optimizer, we need to add the number of shots to the result,
        # so if attribute n_shots is not empty, it is added to the dictionary
        if getattr(self, "n_shots", None) is not None:
            return_dict["n_shots"] = self.n_shots

        return (
            return_dict
            if exclude_keys == []
            else delete_keys_from_dict(return_dict, exclude_keys)
        )

    @classmethod
    def from_dict(
        cls, dictionary: dict, cost_hamiltonian: Union[Hamiltonian, None] = None
    ):
        """
        Creates a Results object from a dictionary (which is the output of the asdict method)
        Parameters
        ----------
        dictionary: `dict`
            The dictionary to create the QAOA Result object from
        Returns
        -------
        `Result`
            The Result object created from the dictionary
        """

        # deepcopy the dictionary, so that the original dictionary is not changed
        dictionary = copy.deepcopy(dictionary)

        # create a new instance of the class
        result = cls.__new__(cls)

        # set the attributes of the new instance, using the dictionary
        for key, value in dictionary.items():
            setattr(result, key, value)

        # if there is an input cost hamiltonian, it is added to the result
        if cost_hamiltonian is not None:
            result.cost_hamiltonian = cost_hamiltonian

        # if the measurement_outcomes are strings, they are converted to complex numbers
        if not isinstance(
            result.optimized["measurement_outcomes"], dict
        ) and isinstance(result.optimized["measurement_outcomes"][0], str):
            for i in range(len(result.optimized["measurement_outcomes"])):
                result.optimized["measurement_outcomes"][i] = complex(
                    result.optimized["measurement_outcomes"][i]
                )

            for i in range(len(result.intermediate["measurement_outcomes"])):
                for j in range(len(result.intermediate["measurement_outcomes"][i])):
                    result.intermediate["measurement_outcomes"][i][j] = complex(
                        result.intermediate["measurement_outcomes"][i][j]
                    )

        # if the measurement_outcomes are complex numbers, the backend is set to QAOABaseBackendStatevector
        if not isinstance(
            result.optimized["measurement_outcomes"], dict
        ) and isinstance(result.optimized["measurement_outcomes"][0], complex):
            pass
        else:
            setattr(result, "_QAOAResult__type_backend", "")

        # return the object
        return result

    @staticmethod
    def get_counts(measurement_outcomes):
        """
        Converts probabilities to counts when the measurement outcomes are a numpy array,
        that is a state vector

        Parameters
        ----------
        measurement_outcomes: `Union[np.array, dict]`
            The measurement outcome as returned by the Logger.
            It can either be a statevector or a count dictionary

        Returns
        -------
        `dict`
            The count dictionary obtained either throught the statevector or
            the actual measurement counts.
        """

        # if isinstance(measurement_outcomes, type(np.array([]))):
        #     measurement_outcomes = qaoa_probabilities(measurement_outcomes)

        return measurement_outcomes

    def plot_cost(
        self, figsize=(10, 8), label="Cost", linestyle="--", color="b", ax=None
    ):
        """
        A simpler helper function to plot the cost associated to a QAOA workflow

        Parameters
        ----------
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        label: `str`
            The label of the cost line, defaults to 'Cost'.
        linestyle: `str`
            The linestyle of the poloit. Defaults to '--'.
        color: `str`
            The color of the line. Defaults to 'b'.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            range(
                1,
                self.evals["number_of_evals"]
                - self.evals["jac_evals"]
                - self.evals["qfim_evals"]
                + 1,
            ),
            self.intermediate["cost"],
            label=label,
            linestyle=linestyle,
            color=color,
        )

        ax.set_ylabel("Cost")
        ax.set_xlabel("Number of function evaluations")
        ax.legend()
        ax.set_title("Cost history")

        return

    def plot_probabilities(
        self,
        n_states_to_keep=None,
        figsize=(10, 8),
        label="Probability distribution",
        color="tab:blue",
        ax=None,
    ):
        """
        Helper function to plot the probabilities corresponding to each basis states
        (with prob != 0) obtained from the optimized result

        Parameters
        ----------
        n_states_to_keep: 'int
            If the user passes a value, the plot will compile with the given value of states.
            Else,  an upper bound will be calculated depending on the
            total size of the measurement outcomes.
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        label: `str`
            The label of the cost line, defaults to 'Probability distribution'.
        color: `str`
            The color of the line. Defaults to 'tab:blue'.
        ax: 'matplotlib.axes._subplots.AxesSubplot'
            Axis on which to plot the graph. Deafults to None
        """

        outcome = self.optimized["measurement_outcomes"]

        # converting to counts dictionary if outcome is statevector
        if isinstance(outcome, type(np.array([]))):
            outcome = self.get_counts(outcome)
            # setting norm to 1 since it might differ slightly for statevectors due to numerical preicision
            norm = np.float64(1)
        else:
            # needed to be able to divide the tuple by 'norm'
            norm = np.float64(sum(outcome.values()))

        # sorting dictionary. adding a callback function to sort by values instead of keys
        # setting reverse = True to be able to obtain the states with highest counts
        outcome_list = sorted(outcome.items(), key=lambda item: item[1], reverse=True)
        states, counts = zip(*outcome_list)

        # normalizing to obtain probabilities
        probs = counts / norm

        # total number of states / number of states with != 0 counts for shot simulators
        total = len(states)

        # number of states that fit without distortion in figure
        upper_bound = 40
        # default fontsize
        font = "medium"

        if n_states_to_keep:
            if n_states_to_keep > total:
                raise ValueError(
                    "n_states_to_keep must be smaller or equal than the total number"
                    f"of states in measurement outcome: {total}"
                )
            else:
                if n_states_to_keep > upper_bound:
                    print("number of states_to_keep exceeds the recommended value")
                    font = "small"

        # if states_to_keep is not given
        else:
            if total > upper_bound:
                n_states_to_keep = upper_bound
            else:
                n_states_to_keep = total

        # formatting labels
        labels = [
            r"$\left|{}\right>$".format(state) for state in states[:n_states_to_keep]
        ]
        labels.append("rest")

        # represent the bar with the addition of all the remaining probabilities
        rest = sum(probs[n_states_to_keep:])

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        colors = [color for _ in range(n_states_to_keep)] + ["xkcd:magenta"]

        ax.bar(labels, np.append(probs[:n_states_to_keep], rest), color=colors)
        ax.set_xlabel("Eigen-State")
        ax.set_ylabel("Probability")
        ax.set_title(label)
        ax.tick_params(axis="x", labelrotation=75, labelsize=font)
        ax.grid(True, axis="y", linestyle="--")

        print("states kept:", n_states_to_keep)
        return

    def plot_n_shots(
        self,
        figsize=(10, 8),
        param_to_plot=None,
        label=None,
        linestyle="--",
        color=None,
        ax=None,
        xlabel="Iterations",
        ylabel="Number of shots",
        title="Evolution of number of shots for gradient estimation",
    ):
        """
        Helper function to plot the evolution of the number of shots used for each evaluation of
        the cost function when computing the gradient.
        It only works for shot adaptive optimizers: cans and icans.
        If cans was used, the number of shots will be the same for each parameter at each iteration.
        If icans was used, the number of shots could be different for each parameter at each iteration.

        Parameters
        ----------
        figsize: `tuple`
            The size of the figure to be plotted. Defaults to (10,8).
        param_to_plot: `list[int]` or `int`
            The parameteres to plot. If None, all parameters will be plotted. Defaults to None.
            If a int is given, only the parameter with that index will be plotted.
            If a list of ints is given, the parameters with those indexes will be plotted.
        label: `list[str]` or `str`
            The label for each parameter. Defaults to Parameter {i}.
            If only one parameter is plot the label can be a string, otherwise it must be a list of strings.
        linestyle: `list[str]` or `str`
            The linestyle for each parameter. Defaults to '--' for all parameters.
            If it is a string all parameters will use it, if it a list of strings the linestyle of
            each parameter will depend on one string of the list.
        color: `list[str]` or `str`
            The color for each parameter. Defaults to None for all parameters (matplotlib will choose the colors).
            If only one parameter is plot the color can be a string, otherwise it must be a list of strings.
        ax: 'matplotlib.axes._subplots.AxesSubplot'
            Axis on which to plot the graph. If none is given, a new figure will be created.

        """

        if ax is None:
            ax = plt.subplots(figsize=figsize)[1]

        # creating a list of parameters to plot
        # if param_to_plot is not given, plot all the parameters
        if param_to_plot is None:
            param_to_plot = list(range(len(self.n_shots[0])))
        # if param_to_plot is a single value, convert to list
        elif type(param_to_plot) == int:
            param_to_plot = [param_to_plot]
        # if param_to_plot is not a list, raise error
        if type(param_to_plot) != list:
            raise ValueError(
                "`param_to_plot` must be a list of integers or a single integer"
            )
        else:
            for param in param_to_plot:
                assert param < len(
                    self.n_shots[0]
                ), f"`param_to_plot` must be a list of integers between 0 and {len(self.n_shots[0]) - 1}"

        # if label is not given, create a list of labels for each parameter (only if there is more than 1 parameter)
        if len(self.n_shots[0]) > 1:
            label = (
                [f"Parameter {i}" for i in param_to_plot] if label is None else label
            )
        else:
            label = ["n. shots per parameter"]

        # if only one parameter is plotted, convert label and color to list if they are string
        if len(param_to_plot) == 1:
            if type(label) == str:
                label = [label]
            if type(color) == str:
                color = [color]

        # if param_top_plot is a list and label or color are not lists, raise error
        if (type(label) != list) or (type(color) != list and color is not None):
            raise TypeError("`label` and `color` must be list of str")
        # if label is a list, check that all the elements are strings
        for lab in label:
            assert type(lab) == str, "`label` must be a list of strings"
        # if color is a list, check that all the elements are strings
        if color is not None:
            for c in color:
                assert type(c) == str, "`color` must be a list of strings"

        # if label and color are lists, check if they have the same length as param_to_plot
        if len(label) != len(param_to_plot) or (
            color is not None and len(color) != len(param_to_plot)
        ):
            raise ValueError(
                f"`param_to_plot`, `label` and `color` must have the same length, \
                    `param_to_plot` is a list of {len(param_to_plot)} elements"
            )

        # linestyle must be a string or a list of strings, if it is a string, convert it to a list of strings
        if type(linestyle) != str and type(linestyle) != list:
            raise TypeError("`linestyle` must be str or list")
        elif type(linestyle) == str:
            linestyle = [linestyle for _ in range(len(param_to_plot))]
        elif len(linestyle) != len(param_to_plot):
            raise ValueError(
                f"`linestyle` must have the same length as param_to_plot \
                    (length of `param_to_plot` is {len(param_to_plot)}), or be a string"
            )
        else:
            for ls in linestyle:
                assert type(ls) == str, "`linestyle` must be a list of strings"

        # plot the evolution of the number of shots for each parameter that is in param_to_plot
        transposed_n_shots = np.array(self.n_shots).T
        for i, values in enumerate([transposed_n_shots[j] for j in param_to_plot]):
            if color is None:
                ax.plot(values, label=label[i], linestyle=linestyle[i])
            else:
                ax.plot(values, label=label[i], linestyle=linestyle[i], color=color[i])

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.set_title(title)

    def lowest_cost_bitstrings(self, n_bitstrings: int = 1) -> dict:
        """
        Find the minimium energy from cost_hamilonian given a set of measurement
        outcoms

        Parameters
        ----------

        n_bitstrings : int
            Number of the lowest energies bistrings to get

        Returns
        -------
        best_results : dict
            Returns a list of bitstring with the lowest values of the cost Hamiltonian.

        """

        if isinstance(self.optimized["measurement_outcomes"], dict):
            measurement_outcomes = self.optimized["measurement_outcomes"]
            solution_bitstring = list(measurement_outcomes.keys())
        elif isinstance(self.optimized["measurement_outcomes"], np.ndarray):
            measurement_outcomes = self.get_counts(
                self.optimized["measurement_outcomes"]
            )
            solution_bitstring = list(measurement_outcomes.keys())
        else:
            raise TypeError(
                f"The measurement outcome {type(self.optimized['measurement_outcomes'])} is not valid."
            )
        energies = [
            bitstring_energy(self.cost_hamiltonian, bitstring)
            for bitstring in solution_bitstring
        ]
        args_sorted = np.argsort(energies)
        if n_bitstrings > len(energies):
            n_bitstrings = len(energies)

        total_shots = sum(measurement_outcomes.values())
        best_results = {
            "solutions_bitstrings": [
                solution_bitstring[args_sorted[ii]] for ii in range(n_bitstrings)
            ],
            "bitstrings_energies": [
                energies[args_sorted[ii]] for ii in range(n_bitstrings)
            ],
            "probabilities": [
                measurement_outcomes[solution_bitstring[args_sorted[ii]]] / total_shots
                for ii in range(n_bitstrings)
            ],
        }
        return best_results


# DEFINE MAIN RUNTIME FUNCTION HERE ###############


def main(
    backend,
    user_messenger,
    qubo_dict: dict,
    p: int,
    n_shots: int,
    circuit_optimization_level: int,
    optimizer_dict: dict,
    **kwargs,
):
    """
    Entry function for the runtime program
    This function will be partially equivalent to q.compile in
    QAOA workflows. Assemble the QAOA objects here for execution
    of the optimization routine

    Parameters
    ----------
    backend: IBMQBackend
        IBMQBackend to run the computation on
    user_messenger
        Messenger relays intermediate information to the user
    qubo_dict: dict
        The QUBO to optimize with QAOA
    p: int
        The depth of the QAOA
    n_shots: int
        Number of shots for circuit evaluation
    circuit_optimization_level: int
        qiskit.transpile level of circuit optimization
    optimizer_dict: dict
        The optimizer dictionary with options supported by OpenQAOA
    """
    publisher = Publisher(user_messenger)
    # create the QUBO object by passing a dictionary of problem class
    qubo = QUBO.from_dict(qubo_dict)

    # initialize OpenQAOA device
    ibm_device = DeviceQiskit(backend)
    ibm_device.check_connection()

    cost_hamil = qubo.hamiltonian

    mixer_hamil = X_mixer_hamiltonian(
        n_qubits=cost_hamil.n_qubits,
    )

    qaoa_descriptor = QAOADescriptor(
        cost_hamil, mixer_hamil, p=p, routing_function=None, device=ibm_device
    )
    variational_params = QAOAVariationalStandardParams.linear_ramp_from_hamiltonian(
        qaoa_descriptor
    )

    oq_backend = QAOAQiskitQPUBackend(
        qaoa_descriptor=qaoa_descriptor,
        device=ibm_device,
        n_shots=n_shots,
        init_hadamard=True,
        prepend_state=kwargs.get("prepend_state", None),
        append_state=kwargs.get("append_state", None),
        qiskit_optimization_level=circuit_optimization_level,
    )

    optimizer = get_optimizer(oq_backend, variational_params, optimizer_dict, publisher)

    # perform the optimization loop
    optimizer.optimize()

    qaoa_result = optimizer.qaoa_result

    # convert the results into JSON serializable
    serialized_qaoa_result = qaoa_result.asdict(complex_to_string=True)

    return serialized_qaoa_result
