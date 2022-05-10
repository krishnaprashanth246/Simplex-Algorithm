#!/usr/bin/env python3

# Linear Optimization Assignment 1

# G Vishal Siva Kumar - CS18BTECH11013
# T Krishna Prashanth - CS18BTECH11045

# In this program, we assume that we do not encounter unbounded polytope 
# and degenerate cases

# Sample Problem:
#	A = [[ 2,  -1],
#	      [ 1,  2],
# 		  [-1, 	1],
#	      [-1,  0],
#	      [ 0, -1]]
#	B = [4,9,3,0,0]
#	C = [2,5]
#	initial feasible point = [2,0]
# 
# Input format for the above problem:
# 	Simplex Algorithm with non-degenerate and rank of matrix equal number of columns
# 	Enter m = 5
# 	Enter n = 2
# 	Enter matrix A in a single line row-wise space separated integers: 
# 	2 -1 1 2 -1 1 -1 0 0 -1
# 	Enter matrix B in a single line row-wise space separated integers: 
# 	4 9 3 0 0
# 	Enter matrix C in a single line row-wise space separated integers: 
# 	2 5
# 	Enter initial feasible point matrix in a single line row-wise space separated integers: 
# 	2 0

# Sample Output:
# 	The maximum for the given Linear Program is at: [1. 4.]
# 	The corresponding objective function value is: 22.0


import numpy as np
import sys
from numpy import linalg as la


def SimplexAlgorithm(A, B, C, initPoint):
	
	m = A.shape[0]
	# get the initial feasible point
	currPoint = initPoint

	for k in range(1000*m*m):
		# rows in A where the current point satisfies equality
		trs = []
		for i in range(m):
			if  C.shape[0] == len(trs):
				break
			if abs((A[i]@currPoint) - B[i]) < 1e-9:
				trs.append(i)

		Atr = A[trs]

		# get alpha where dot product of alpha and Atr is equal to C
		alpha = C@(la.inv(Atr))

		# check if there are any negative elements in alpha
		negidxlist = np.where(alpha<0)[0]
		
		# Current point is the optimal point if all alphas are positive
		if len(negidxlist) == 0:
			return currPoint
		negidx = negidxlist[0]

		# the direction vector 
		directv = -la.inv(Atr)[:, negidx]

		min_t = None
		i = 0
		while i < m:
			if (i not in trs) and (A[i]@directv > 0):
				slack = (B[i] - A[i]@currPoint)
				curr_t = slack/(A[i]@directv)
				
				if min_t is None or curr_t < min_t:
					min_t = curr_t
			i = i + 1

		currPoint = currPoint + directv*min_t



if __name__ == '__main__':
	print("Simplex Algorithm with bounded polytope, non-degenerate and rank of matrix equal number of columns")
	m = int(input("Enter m = "))
	n = int(input("Enter n = "))

	tempA = np.array(input("Enter matrix A in a single line row-wise space separated integers: \n").split(' '),float)
	A = tempA.reshape(m,n)
	B = np.array(input("Enter matrix B in a single line row-wise space separated integers: \n").split(' '),float)
	C = np.array(input("Enter matrix C in a single line row-wise space separated integers: \n").split(' '),float)

	initfeasible = np.array(input("Enter initial feasible point matrix in a single line row-wise space separated integers: \n").split(' '),float)

	solution = SimplexAlgorithm(A, B, C, initfeasible)
	print("The maximum for the given Linear Program is at: {}".format(str(solution)))
	print("The corresponding objective function value is: {}".format(C@solution))