package lap

import (
	"math"
	"sort"
)

// JacobiSVD computes the Jacobi SVD of A, returning U, Sigma, V^T
func JacobiSVD(A *DenseM) (sigma *DenseV) {
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
	// populate V as the identity matrix
	G := NewDenseMatrix(2, 2, nil)
	V := NewDenseMatrix(nrow, nrow, nil)
	V.Copy(Eye(nrow))
	var i, j int
	tolsigma := tol * Norm(A, 2)
	Aux := NewDenseMatrix(ncol, 2, nil)
	for rots >= 1 {
		i++
		rots = 0
		for p := 0; p < ncol; p++ {
			colp := A.ColView(p)
			for q := p + 1; q < ncol; q++ {
				sp := fsigma[p]
				sq := fsigma[q]
				spq := sp * sq
				colq := A.ColView(q)
				beta := Dot(colp, colq)
				if spq > tolsigma && math.Abs(beta) >= tol*math.Sqrt(spq) {
					j++
					rots++
					t := G.jacobi(sp, beta, sq)
					fsigma[p] = sp - beta*t
					fsigma[q] = sq + beta*t
					Aux.Mul(Slice(A, nil, []int{p, q}), G)
					colp.CopyVec(Aux.ColView(0))
					colq.CopyVec(Aux.ColView(1))
				}
			}
		}
	}

	// Post Processing
	sort.Float64s(fsigma)
	for k := 0; k < ncol; k++ {
		s := fsigma[k]
		if s == 0 {
			for i := k; i < len(fsigma); i++ {
				fsigma[i] = 0
			}
			break
		}
		fsigma[k] = math.Sqrt(s)
	}
	return NewDenseVector(ncol, fsigma)
}

// The Jacobi rotation is a plane unitary similarity transformation:
//
//	[ c  s ]T [ alpha  beta ]  [ c  s ]  =  [ l1  0 ]
//	[-s  c ]  [ beta  gamma ]  [-s  c ]  =  [ 0  l2 ]
//
// where G = [c, s; -s, c]
func (G *DenseM) jacobi(alpha, beta, gamma float64) (t float64) {
	if r, c := G.Dims(); r != 2 && c != 2 {
		panic(ErrDim)
	}
	var c, s float64
	if beta != 0 {
		tau := (gamma - alpha) / (2 * beta)
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

// MatInvertSquare inverts square matrix A of dimension nxn, storing the result in out.
// scratch must be n x 2n or nil in which case the slice is allocated temporarily.
//
// Gauss-Jordan elimination is used to perform the inversion, A must be non-singular.
func (out *DenseM) invertSquare(A Matrix, scratchSlice []float64) error {
	n, c := A.Dims()
	if n != c {
		return ErrDim
	}
	n2 := 2 * n
	if out.data == nil {
		*out = *NewDenseMatrix(n, n, nil)
	}
	var scratchZeroed bool
	if scratchSlice == nil {
		scratchZeroed = true
		scratchSlice = make([]float64, n*n2)
	} else if len(scratchSlice) < n*n2 {
		return ErrDim
	}
	scratch := NewDenseMatrix(n, n2, scratchSlice)

	// make scratch into the augmenting identity matrix
	for i := 0; i < n; i++ {
		ridx := i * scratch.stride
		for j := 0; j < n2; j++ {
			if j < n {
				scratch.data[ridx+j] = A.At(i, j)
			}
			if j == i+n {
				scratch.data[ridx+j] = 1
			} else if !scratchZeroed {
				// would not need this if we alloced (guaranteed zero)
				// but to be safe, zero here
				scratch.data[ridx+j] = 0
			}
		}
	}
	// exchange rows of the matrix, bottom-up
	for i := n - 1; i > 0; i-- {
		if scratch.At(i-1, 0) < scratch.At(i, 0) {
			scratch.SwapRows(i, i-1)
		}
	}

	// replace each row by sum of itself and a constant times another row
	for i := 0; i < n; i++ {
		// ic := i * scratch.c
		for j := 0; j < n; j++ {
			if i != j {
				tmp := scratch.At(j, i) / scratch.At(i, i)
				for k := 0; k < n2; k++ {
					result := scratch.At(j, k) - scratch.At(i, k)*tmp
					scratch.Set(j, k, result)
				}
			}
		}
	}
	const eps = 1e-16
	// mul each row by a nonzero integer and divide each row by the diagonal
	for i := 0; i < n; i++ {
		tmp := scratch.At(i, i)
		if math.Abs(tmp) < eps {
			return ErrSingular
		}
		for j := 0; j < n2; j++ {
			scratch.Set(i, j, scratch.At(i, j)/tmp)
		}
	}

	// scratch now contains the inverse of input in its lefthand half
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out.Set(i, j, scratch.At(i, j))
		}
	}
	return nil
}
