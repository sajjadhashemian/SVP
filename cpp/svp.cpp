#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "basis.h"
#include "decision_svp.h"
namespace py = pybind11;


// Pybind11 bindings
PYBIND11_MODULE(svp, m)
{
	m.doc() = "Decision-SVP module using Pybind11";

	m.def("decision_svp", &decisionSVP,
		  py::arg("B"),
		  py::arg("R"),
		  py::arg("sigma"),
		  py::arg("num_samples"),
		  py::arg("seed"),
		  "Solve the Decision-SVP problem");

	m.def("basis_reduction", &basis_reduction,
		  py::arg("B"),
		  py::arg("R"),
		  py::arg("sigma"),
		  py::arg("num_samples"),
		  py::arg("seed"),
		  "Randomized Lattice Basis Reduction");
}