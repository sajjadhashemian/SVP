#ifndef INCLUDE
#define INCLUDE
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <limits>
#include <sstream> // <-- Added
using namespace Eigen;
using namespace std;

template <typename Derived>
std::string get_shape(const EigenBase<Derived> &x)
{
	std::ostringstream oss;
	oss << "(" << x.rows() << ", " << x.cols() << ")";
	return oss.str();
}

// Helper function to generate normally distributed random numbers
double normalRandom(double mean, double stddev)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<> dis(mean, stddev);
	return dis(gen);
}

// Function to sample vectors and compute their norms
std::pair<VectorXd, VectorXd> sample(const MatrixXd &B, const MatrixXd &B_pinv, double R, double sigma)
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

	return {z, v1};
}

#endif