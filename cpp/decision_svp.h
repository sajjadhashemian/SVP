#ifndef COMMON
#define COMMON
#include "common.h"
#endif

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

	for (long long counter = 0; counter < num_samples; ++counter)
	{
		auto [z, v] = sample(B, B_pinv, R, sigma);

		if (z.norm() > 1e-5 && z.norm() < len_vector + 0.0001)
		{
			len_vector = z.norm();
			short_vector = z;
			index_vector = v;
			if (len_vector <= R + 0.0001)
			{
				// for(auto x: v)
				// 	cout<<x<<", ";cout<<endl;
				return {index_vector, len_vector, counter, true};
			}
		}
	}
	// for (auto x : index_vector)
	// 	cout << x << "; ";
	// cout << endl;
	return {index_vector, len_vector, num_samples, false};
}
