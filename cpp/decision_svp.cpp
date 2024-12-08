#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
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
std::pair<VectorXd, double> sample(const MatrixXd &B, const MatrixXd &B_pinv, int n, double R, double sigma)
{
	double r = normalRandom(R, sigma);
	VectorXd direction(n);
	for (int i = 0; i < n; ++i)
	{
		direction(i) = normalRandom(0, 1);
	}
	direction.normalize();
	VectorXd v = r * direction;
	VectorXd z = B * ((B_pinv * v).array().round().matrix());
	double norm_z = z.norm();
	return {z, norm_z};
}

// Main function for Decision-SVP
std::tuple<VectorXd, double, long long, bool> decisionSVP(const MatrixXd &B, double R, double sigma, long long num_samples, int seed)
{
	int n = B.rows();
	// int m = B.cols();
	VectorXd short_vector = VectorXd::Zero(n);
	double len_vector = std::numeric_limits<double>::infinity();

	// Compute pseudo-inverse of B
	MatrixXd B_pinv = B.completeOrthogonalDecomposition().pseudoInverse();

	// Seed the random number generator
	std::srand(seed);

	for (long long counter = 0; counter < num_samples; ++counter)
	{
		auto [z, norm_z] = sample(B, B_pinv, n, R, sigma);
		if (norm_z > 1e-5 && norm_z < len_vector + 0.0001)
		{
			len_vector = norm_z;
			short_vector = z;
			if (len_vector <= R + 0.1)
			{
				return {short_vector, len_vector, counter, true};
			}
		}
	}
	return {short_vector, len_vector, num_samples, false};
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
