import numpy as np
from scipy.stats import pearsonr

EPS = 1e-1
N = 10
k = 3

x = np.random.random(N)
y = np.random.random(N)

print pearsonr(x,y)[0]

for step in range(100):
	if step % 10 == 0:
		print pearsonr(x,y)[0]

	y_no_mean = y - np.mean(y)
	x_no_mean = x - np.mean(x)

	sigma_x = np.std(x) * np.sqrt(N)
	sigma_y = np.std(y) * np.sqrt(N)

	grad_k = ((y_no_mean * sigma_x) - (np.dot(x_no_mean, y_no_mean) * x_no_mean / sigma_x)) / ((sigma_x**2) * sigma_y)

	x += EPS*grad_k

