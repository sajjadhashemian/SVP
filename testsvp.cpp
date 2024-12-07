#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <limits>

using namespace Eigen;
using namespace std;

double normalRandom(double mean, double stddev)
{
	static random_device rd;
	static mt19937 gen(rd());
	normal_distribution<> dis(mean, stddev);
	return dis(gen);
}

pair<VectorXd, double> sample(const MatrixXd &B, const MatrixXd &B_pinv, int n, double R, double sigma)
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

tuple<VectorXd, double, int, bool> decisionSVP(const MatrixXd &B, double R, double sigma, int num_samples, int seed)
{
	int n = B.rows();
	int m = B.cols();
	VectorXd short_vector = VectorXd::Zero(n);
	double len_vector = numeric_limits<double>::infinity();

	MatrixXd B_pinv = B.completeOrthogonalDecomposition().pseudoInverse();

	srand(seed);

	for (int counter = 0; counter < num_samples; counter++)
	{
		auto [z, norm_z] = sample(B, B_pinv, n, R, sigma);
		if (norm_z > 1e-5 && norm_z < len_vector+1e5)
		{
			len_vector = norm_z;
			short_vector = z;
			if (len_vector < R + 1e-5)
			{
				return {short_vector, len_vector, counter+1, true};
			}
		}
	}
	return {short_vector, len_vector, num_samples, false};
}

int main()
{
	MatrixXd B(2, 2);
	B << 1, 100000,
		1, 100001;
	double R = 1.0;
	double sigma = 1.0;
	int num_samples = 1<<30;
	int seed = 1337;

	auto [short_vector, len_vector, counter, found] = decisionSVP(B, R, sigma, num_samples, seed);

	cout << "Short vector: " << short_vector.transpose() << endl;
	cout << "Length of short vector: " << len_vector << endl;
	cout << "Number of samples: " << counter << endl;
	cout << "Found Shortest: " << (found ? "Yes" : "No") << endl;

	return 0;
}
