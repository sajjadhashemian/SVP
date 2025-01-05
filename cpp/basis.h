#ifndef COMMON
#define COMMON
#include "common.h"
#endif

// Function to compute the rank of a matrix
int matrix_rank(const MatrixXd &M, double tol = 1e-8)
{
	Eigen::JacobiSVD<MatrixXd> svd(M);
	return (svd.singularValues().array() > tol).count();
}

// Function to compute the angle between two vectors
double angle_between(const VectorXd &a, const VectorXd &b)
{
	return std::acos(a.dot(b) / (a.norm() * b.norm()));
}

// Main function for basis reduction
std::tuple<MatrixXd, long long, bool> basis_reduction(const MatrixXd &B_init, double R, double sigma, long long num_samples, int seed)
{
	int n = B_init.rows();
	MatrixXd B = B_init;
	MatrixXd B_pinv_init = B.completeOrthogonalDecomposition().pseudoInverse();

	double len_vector = std::numeric_limits<double>::infinity();

	MatrixXd basis = B_init;
	double sum_angle = 0.0;
	for (int i = 0; i < B.cols(); i++)
		for (int j = i + 1; j < B.cols(); j++)
			sum_angle += angle_between(B.col(i), B.col(j));
	MatrixXd B_pinv = B.completeOrthogonalDecomposition().pseudoInverse();

	// Seed the random number generator
	std::srand(seed);

	for (long long counter = 0; counter < num_samples; ++counter)
	{
		// Sample a new vector Z
		auto [z, v] = sample(B, B_pinv, R, sigma);

		if (z.norm() < 1e-5)
			continue;

		if (z.norm() < R + 0.0001)
		{
			B.row(0) = v;
			return {B, counter, true};
		}

		// Evaluate replacing a basis vector to maximize angular separation
		bool is_changed = false;
		double max_new_angle = -1.0;
		int replace_index = -1;

		for (size_t i = 0; i < basis.cols(); ++i)
		{
			double temp_sum_angle = sum_angle;
			for (size_t j = 0; j < basis.cols(); ++j)
			{
				if (i != j)
					temp_sum_angle += angle_between(z, basis.col(i)) - angle_between(basis.col(i), basis.col(j));
			}
			if (temp_sum_angle > sum_angle+1e-5)
			{
				MatrixXd M = basis;
				M.col(i) = z;
				if (matrix_rank(M) == matrix_rank(basis))
				{
					max_new_angle = temp_sum_angle;
					replace_index = i;
				}
			}
		}

		if (replace_index != -1 && max_new_angle > sum_angle)
		{
			// Replace the vector in the basis
			// cout << "shapes: " << basis.row(0).size() << ", " << basis.col(0).size() << " " << z.size() << endl;
			basis.col(replace_index) = z;
			is_changed = true;
		}

		if (is_changed == true)
		{
			B = basis;
			B_pinv = B.completeOrthogonalDecomposition().pseudoInverse();
			sum_angle = 0.0;
			for (int i = 0; i < B.cols(); i++)
			{
				for (int j = i + 1; j < B.cols(); j++)
				{
					sum_angle += angle_between(B.col(i), B.col(j));
				}
			}
		}
	}

	return {B, 0, false};
}
