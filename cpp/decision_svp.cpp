#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <limits>

namespace py = pybind11;
using namespace Eigen;

// Helper function to generate normally distributed random numbers
double normalRandom(double mean, double stddev)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<> dis(mean, stddev);
	return dis(gen);
}

// Function to sample vectors and compute their norms
std::pair<std::pair<VectorXd, double>, VectorXd> sample(const MatrixXd &B, const MatrixXd &B_pinv, double R, double sigma)
{
	double r = normalRandom(R, sigma);
	int n = B_pinv.cols();
	// std::cout << "Shape B: " << B.rows() << ", " <<B.cols()<< std::endl;
	// std::cout << "Shape B^-1: " << B_pinv.rows() << ", " << B_pinv.cols() << std::endl;

	VectorXd direction(n);
	for (int i = 0; i < n; ++i)
	{
		direction(i) = normalRandom(0, 1);
	}
	direction.normalize();
	VectorXd v = r * direction;
	// std::cout << "Shape v: " << v.rows() << ", " << v.cols() << std::endl;

	VectorXd v1 = (B_pinv * v).array().round().matrix();
	// std::cout << "Shape v^-1: " << v1.rows() << ", " << v1.cols() << std::endl;

	VectorXd z = B * v1;
	// std::cout << "Shape z: " << z.rows() << ", " << z.cols() << std::endl;

	double norm_z = z.norm();
	return {{z, norm_z}, v1};
}

// Main function for Decision-SVP
std::tuple<VectorXd, double, long long, bool> decisionSVP(const MatrixXd &B, double R, double sigma, long long num_samples, int seed)
{
	int n = B.rows();
	// int m = B.cols();
	VectorXd short_vector = VectorXd::Zero(n);
	VectorXd index_vector = VectorXd::Zero(n);
	double len_vector = std::numeric_limits<double>::infinity();

	// Compute pseudo-inverse of B
	MatrixXd B_pinv = B.completeOrthogonalDecomposition().pseudoInverse();

	// Seed the random number generator
	std::srand(seed);
	// std::cout<<"-----SEED: "<<seed<<std::endl;

	for (long long counter = 0; counter < num_samples; ++counter)
	{
		auto [Z, v] = sample(B, B_pinv, R, sigma);
		auto [z, norm_z] = Z;
		if (norm_z > 1e-5 && norm_z < len_vector + 0.0001)
		{
			len_vector = norm_z;
			short_vector = z;
			index_vector = v;
			if (len_vector <= R + 0.0001)
			{
				return {index_vector, len_vector, counter, true};
			}
		}
	}
	return {index_vector, len_vector, num_samples, false};
}

// Pybind11 bindings
PYBIND11_MODULE(decision_svp, m)
{
	m.doc() = "Decision-SVP module using Pybind11";

	m.def("decision_svp", &decisionSVP,
		  py::arg("B"),
		  py::arg("R"),
		  py::arg("sigma"),
		  py::arg("num_samples"),
		  py::arg("seed"),
		  "Solve the Decision-SVP problem");
}