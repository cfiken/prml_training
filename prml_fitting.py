import numpy as np
import scipy
import matplotlib.pyplot as plt

##############
# parameters #
##############

M = 9	# dimension
N = 50	# number of data

noise_variance = 0.3


def makeData(x):
	return np.sin(2*np.pi*x) + np.random.randn(len(x))*noise_variance

def y(x,w):
	target = np.zeros(len(x))
	for i in range(M+1):
		target += x**i * w[i]
	return target

def estimate(x, t):
	A = np.zeros((M+1,M+1))
	T = np.zeros(M+1)
	for i in range(M+1):
		T[i] = ((x**i) * t).sum()
		for j in range(M+1):
			A[i,j] = (x**(i+j)).sum()

	w = np.linalg.solve(A, T)
	return w

def main():
	# training data
	x = np.linspace(0,1,N)
	t = makeData(x)
	w = estimate(x,t)
	
	plt.scatter(x,t)

	x_line = np.linspace(0,1,201)
	y_line = np.sin(2*np.pi*x_line)
	y_estimate = y(x_line,w)

	plt.plot(x_line, y_estimate, color='red',label='estimated')
	plt.plot(x_line, y_line, color='green', label='sin(2*pi*x)')
	plt.legend()
	plt.show()

main()