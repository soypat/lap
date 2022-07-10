package lap

import (
	"errors"
	"math"
)

var (
	ErrSingular    = errors.New("matrix is singular to working precision")
	ErrAliasedData = errors.New("aliased data")
	ErrRowAccess   = errors.New("bad row access")
	ErrColAccess   = errors.New("bad column access")
	ErrDim         = errors.New("bad dimension")
)

type Matrix interface {
	At(i, j int) float64
	Dims() (r, c int)
}

// DenseM represents a row major storage matrix.
type DenseM struct {
	data []float64
	r, c int
}

// Dims returns the dimensions of the matrix.
func (d DenseM) Dims() (int, int) { return d.r, d.c }

// At returns d's element at ith row, jth column.
func (d DenseM) At(i, j int) float64 {
	if i < 0 || i >= d.r {
		panic(ErrRowAccess)
	} else if j < 0 || j >= d.c {
		panic(ErrColAccess)
	}
	return d.data[i*d.c+j]
}

// Set sets d's element at ith row, jth column to v.
func (d *DenseM) Set(i, j int, v float64) {
	if i < 0 || i >= d.r {
		panic(ErrRowAccess)
	} else if j < 0 || j >= d.c {
		panic(ErrColAccess)
	}
	d.data[i*d.c+j] = v
}

// CopyFrom produces a copy of A with no overlapping memory.
// If the receiver is not initialized then the backing array is allocated
// automatically.
func (d *DenseM) CopyFrom(A Matrix) {
	r, c := A.Dims()
	if d.data == nil {
		*d = NewDenseMatrix(r, c, nil)
	}
	if r != d.r || c != d.c {
		panic(ErrDim)
	}
	for i := 0; i < d.r; i++ {
		for j := 0; j < d.c; j++ {
			d.data[i*d.c+j] = A.At(i, j)
		}
	}
}

// NewDenseMatrix produces a new (rxc) matrix backed by contiguous data.
// this function produces superior memory access patterns and prevents the rows
// of the output from being scattered in memory.
//
// data may be nil, in which case an array of zeros is returned
func NewDenseMatrix(r, c int, data []float64) (d DenseM) {
	if data == nil {
		data = make([]float64, r*c)
	}
	return DenseM{
		data: data,
		r:    r,
		c:    c,
	}
}

type eye int

func (e eye) Dims() (int, int) { return int(e), int(e) }
func (e eye) At(i, j int) float64 {
	if i < 0 || i > int(e) {
		panic(ErrRowAccess)
	}
	if j < 0 || j > int(e) {
		panic(ErrColAccess)
	}
	if i == j {
		return 1
	}
	return 0
}

// Eye is the square identity matrix of size N
func Eye(n int) Matrix {
	return eye(n)
}

// Norm calculates the norm of the matrix. Only Frobenius is implemented.
//  1 - The maximum absolute column sum
//  2 - The Frobenius norm, the square root of the sum of the squares of the elements
//  Inf - The maximum absolute row sum
// This implements gonum's Normer interface.
func (d DenseM) Norm(norm float64) float64 {
	if norm != 2 {
		panic("not implemented")
	}
	var out float64
	for i := 0; i < d.r; i++ {
		for j := 0; j < d.c; j++ {
			g := d.At(i, j)
			out += g * g
		}
	}
	return math.Sqrt(out)
}

type Transpose struct {
	m Matrix
}

func (t Transpose) At(i, j int) float64 {
	return t.m.At(j, i)
}

func (t Transpose) Dims() (int, int) {
	c, r := t.m.Dims()
	return r, c
}

// T returns the implicit transpose of A without copying.
func T(A Matrix) Matrix {
	if t, ok := A.(Transpose); ok {
		// If matrix is of underlying transpose type, we untranspose
		// by unwrapping the transpose type
		return t.m
	}
	return Transpose{m: A}
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
		*out = NewDenseMatrix(n, n, nil)
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
		ridx := i * scratch.c
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

// MatMul computes the matrix-matrix product C = AB for (nxm) matrix A and (mxp)
// matrix B, storing the result in (nxp) matrix C.
func (C *DenseM) Mul(A, B Matrix) {
	n, m := A.Dims()
	mB, p := B.Dims()
	if C.data == nil {
		*C = NewDenseMatrix(n, p, nil)
	}
	nC, pC := C.Dims()
	if m != mB || nC != n || pC != p {
		panic(ErrDim)
	}
	if aliasedData(C, A) || aliasedData(C, B) {
		panic(ErrAliasedData)
	}
	for i := 0; i < n; i++ {
		ridx := i * nC
		for j := 0; j < p; j++ {
			tmp := 0.0
			for k := 0; k < m; k++ {
				tmp += A.At(i, k) * B.At(k, j)
			}
			C.data[ridx+j] = tmp
		}
	}
}

// Sub stores the elementwise addition A+B in C.
func (C DenseM) Add(A, B Matrix) {
	r, c := C.Dims()
	rA, cA := A.Dims()
	rB, cB := B.Dims()
	if rA != r || rB != r || cA != c || cB != c {
		panic(ErrDim)
	}
	for i := 0; i < r; i++ {
		ridx := i * C.c
		for j := 0; j < c; j++ {
			C.data[ridx+j] = A.At(i, j) + B.At(i, j)
		}
	}
}

// Sub stores the elementwise difference A-B in C.
func (C DenseM) Sub(A, B Matrix) {
	r, c := C.Dims()
	rA, cA := A.Dims()
	rB, cB := B.Dims()
	if rA != r || rB != r || cA != c || cB != c {
		panic(ErrDim)
	}
	for i := 0; i < r; i++ {
		ridx := i * C.c
		for j := 0; j < c; j++ {
			C.data[ridx+j] = A.At(i, j) - B.At(i, j)
		}
	}
}

// SwapRows swaps rows i and j of A in-place.
func (A DenseM) SwapRows(i, j int) {
	for k := 0; k < A.c; k++ {
		A.data[i*A.c+k], A.data[j*A.c+k] = A.data[j*A.c+k], A.data[i*A.c+k]
	}
}

func (A DenseM) SwapCols(i, j int) {
	for k := 0; k < A.r; k++ {
		A.data[k*A.c+i], A.data[k*A.c+j] = A.data[k*A.c+j], A.data[k*A.c+i]
	}
}

func (A DenseM) RowView(i int) Vector {
	if i >= A.r || i < 0 {
		panic(ErrRowAccess)
	}
	return DenseV{
		data: A.data[i*A.c : (i+1)*A.c],
	}
}

func (A DenseM) ColView(j int) Vector {
	if j >= A.c || j < 0 {
		panic(ErrColAccess)
	}
	return DenseV{
		data:        A.data[j:],
		incMinusOne: A.c - 1,
	}
}
