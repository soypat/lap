package lap

import "math"

// JacobiSVD computes the Jacobi SVD of A, returning U, Sigma, V^T
func JacobiSVD(A DenseM) (U, S, Vtrans Matrix) {
	const tol = 1e-14
	var rots int = 1
	nrow, ncol := A.Dims()
	fsigma := make([]float64, ncol)
	// populate sigma with the columnwise squared sum of A
	for i := 0; i < nrow; i++ {
		for j := 0; j < ncol; j++ {
			v := A.At(i, j)
			fsigma[j] += v * v
		}
	}
	sigma := NewDenseVector(ncol, fsigma)
	// populate V as the identity matrix
	G := NewDenseMatrix(2, 2, nil)
	V := Eye(nrow)
	var i, j, iters int
	tolsigma := tol * Norm(A, 2)
	for rots >= 1 {
		i++
		rots = 0
		for p := 0; p < ncol; p++ {
			sp := fsigma[p]
			// No derijk
			// k := Max(sigma)
			// k = k + p - 1
			// if k != p {
			// 	fsigma[k], fsigma[p] = fsigma[p], fsigma[k]
			// 	// want to express the matlab syntax A(:, [k, p]) = A(:, [p, k])
			// 	// this swaps two columns of A
			// 	// do the same in V

			// 	// https://github.com/zlliang/jacobi-svd/blob/master/jacobi_svd.m
			// }
			for q := p; q < ncol; q++ {
				sq := fsigma[q]
				spq := sp * fsigma[q]
				beta := Dot(A.ColView(p), A.ColView(q))
				if spq > tolsigma && math.Abs(beta) >= tol*math.Sqrt(spq) {
					j++
					rots++
					t := G.jacobi(sp, beta, sq)
					fsigma[p] = sp - beta*t
					fsigma[q] = sq + beta*t
					pr := A.ColView(p)
				}
			}

		}
	}
}

// The Jacobi rotation is a plane unitary similarity transformation:
//  [ c  s ]T [ alpha  beta ]  [ c  s ]  =  [ l1  0 ]
//  [-s  c ]  [ beta  gamma ]  [-s  c ]  =  [ 0  l2 ]
// where G = [c, s; -s, c]
func (G *DenseM) jacobi(alpha, beta, gamma float64) (t float64) {
	if r, c := G.Dims(); r != 2 && c != 2 {
		panic(ErrDim)
	}
	var c, s float64
	if beta != 0 {
		tau := gamma - alpha/(2*beta)
		if tau >= 0 {
			t = 1 / (tau + math.Sqrt(1+tau*tau))
		} else {
			t = -1 / (-tau + math.Sqrt(1+tau*tau))
		}
		c = 1 / math.Sqrt(1+t*t)
		s = t * c
	} else {
		c = 1
	}
	G.Set(0, 0, c)
	G.Set(0, 1, s)
	G.Set(1, 0, -s)
	G.Set(1, 1, c)
	return t
}
