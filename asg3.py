#!/usr/bin/env python3

# Linear Optimization Assignment 3
# G Vishal Siva Kumar - CS18BTECH11013
# T Krishna Prashanth - CS18BTECH11045

# Sample Problem:
#	A = [[ 1,  3],
#	      [ 2,  -1],
#	      [-1,  0],
#	      [ 0, -1]]
#	B = [15,12,0,0]
#	C = [1,2]
# 
# Input format for the above problem:
#	Simplex Algorithm with rank of matrix equal number of columns
# 	Enter m = 4
# 	Enter n = 2
# 	Enter matrix A in a single line row-wise space separated integers: 
# 	1 3 2 -1 -1 0 0 -1
# 	Enter matrix B in a single line row-wise space separated integers: 
# 	15 12 0 0
# 	Enter matrix C in a single line row-wise space separated integers: 
# 	1 2


# Sample Output:
# 	The maximum for the given Linear Program is at: [7.28571755 2.57143075]
# 	The corresponding objective function value is: 12.428579061691174


import numpy as np
import time
import sys
from numpy import linalg as la


def degenerateSimplex(A, B, C):

	A, B, C = removeDegeneracy(A,B,C)
	return degenerateSimplexHelper(A, B, C)

def removeDegeneracy(A,B,C):

	for iters in range(1000):
		i = A.shape[0]-A.shape[1]
		tempB = B
		tempB[:i] = B[:i] + np.random.uniform(1e-6,1e-5,size=i)

		X = getFeasiblePoint(A,tempB,C)
		Z = (A@X) - tempB

		ics =  np.where(np.abs(Z)<1e-6)[0]
		if len(ics) == A.shape[1]:
			return A,tempB,C
	
	while True:
		i = A.shape[0]-A.shape[1]
		tempB = B
		tempB[:i] = B[:i] + np.random.uniform(0.1,10,size=i)

		X = getFeasiblePoint(A,tempB,C)
		Z = (A@X) - tempB

		ics =  np.where(np.abs(Z)<1e-6)[0]
		if len(ics) == A.shape[1]:
			return A,tempB,C

def getFeasiblePoint(A, B, C):
	ics =  np.where(B<0)[0]
	if len(ics) == 0:
		return np.zeros(C.shape[0])

	
	n = A.shape[1]
	m = A.shape[0]
	
	for i in range(100*m*m):
		# get random n rows
		randRows = np.random.choice(m, n)
		Bm = B[randRows]
		Am = A[randRows]
		if la.matrix_rank(Am) != Am.shape[0]:
			continue
		possiblePoint = (la.inv(Am))@(Bm)

		trs = []
		acceptable = True
		j = 0
		while j < m:
			if abs((A[j]@(possiblePoint)) - B[j]) < 1e-9:
				trs.append(j)
			elif A[j]@(possiblePoint) > B[j]:
				acceptable = False
				break
			j = j + 1
		if len(trs) < n:
			acceptable = False
		if acceptable:
			return possiblePoint
	print("Problem is infeasible")
	sys.exit()

# for degenerate questions we use this call to jump from one vertex to another
def degenerateSimplexHelper(A, B, C):

	m = A.shape[0]
	# get the initial feasible point
	currPoint = getFeasiblePoint(A, B, C)

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
		if min_t is None:
			print("The polytope for this problem is unbounded")
			sys.exit()
		currPoint = currPoint + directv*min_t

	print("The polytope for this problem is unbounded")
	sys.exit()


if __name__ == '__main__':
	print("Simplex Algorithm with rank of matrix equal number of columns")
	m = int(input("Enter m = "))
	n = int(input("Enter n = "))

	tempA = np.array(input("Enter matrix A in a single line row-wise space separated integers: \n").split(' '),float)
	A = tempA.reshape(m,n)
	B = np.array(input("Enter matrix B in a single line row-wise space separated integers: \n").split(' '),float)
	C = np.array(input("Enter matrix C in a single line row-wise space separated integers: \n").split(' '),float)

	solution = degenerateSimplex(A, B, C)
	print("The maximum for the given Linear Program is at: {}".format(str(solution)))
	print("The corresponding objective function value is: {}".format(C@solution))