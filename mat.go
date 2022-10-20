package lap

import (
	"errors"
	"fmt"
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
	data   []float64
	stride int
	r, c   int
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
	return d.data[i*d.stride+j]
}

// Set sets d's element at ith row, jth column to v.
func (d *DenseM) Set(i, j int, v float64) {
	if i < 0 || i >= d.r {
		panic(ErrRowAccess)
	} else if j < 0 || j >= d.c {
		panic(ErrColAccess)
	}
	d.data[i*d.stride+j] = v
}

// Copy produces a copy of A with no overlapping memory.
// If the receiver is not initialized then the backing array is allocated
// automatically.
func (d *DenseM) Copy(A Matrix) {
	r, c := A.Dims()
	if d.data == nil {
		*d = NewDenseMatrix(r, c, nil)
	}
	if r != d.r || c != d.c {
		panic(ErrDim)
	}
	for i := 0; i < d.r; i++ {
		for j := 0; j < d.c; j++ {
			d.data[i*d.stride+j] = A.At(i, j)
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
		data:   data,
		r:      r,
		c:      c,
		stride: c,
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

// Slice returns a new Matrix that shares backing data with the receiver.
// The returned matrix starts at {i,j} of the receiver and extends k-i rows
// and l-j columns. The final row in the resulting matrix is k-1 and the
// final column is l-1.
// Slice panics with ErrIndexOutOfRange if the slice is outside the capacity
// of the receiver.
func (d DenseM) Slice(i, k, j, l int) DenseM {
	mr, mc := d.Dims()
	if k <= i || l <= j {
		// Common error or group with below?
		panic(ErrDim)
	}
	if i < 0 || mr <= i || j < 0 || mc <= j || mr < k || mc < l {
		panic(ErrDim)
	}
	d.data = d.data[i*d.stride+j : (k-1)*d.stride+l]
	d.r = k - i
	d.c = l - j
	return d
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
		ridx := i * C.stride
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
func (C *DenseM) Add(A, B Matrix) {
	rA, cA := A.Dims()
	rB, cB := B.Dims()
	if C.data == nil {
		*C = NewDenseMatrix(rA, cA, nil)
	}
	r, c := C.Dims()
	if rA != r || rB != r || cA != c || cB != c {
		panic(ErrDim)
	}
	for i := 0; i < r; i++ {
		ridx := i * C.stride
		for j := 0; j < c; j++ {
			C.data[ridx+j] = A.At(i, j) + B.At(i, j)
		}
	}
}

// Sub stores the elementwise difference A-B in C.
func (C *DenseM) Sub(A, B Matrix) {
	rA, cA := A.Dims()
	rB, cB := B.Dims()
	if C.data == nil {
		*C = NewDenseMatrix(rA, cA, nil)
	}
	r, c := C.Dims()
	if rA != r || rB != r || cA != c || cB != c {
		panic(ErrDim)
	}
	for i := 0; i < r; i++ {
		ridx := i * C.stride
		for j := 0; j < c; j++ {
			C.data[ridx+j] = A.At(i, j) - B.At(i, j)
		}
	}
}

// Scale multiplies the elements of A by f, placing the result in the receiver.
func (C DenseM) Scale(f float64, A Matrix) {
	r, c := C.Dims()
	rA, cA := A.Dims()
	if rA != r || cA != c {
		panic(ErrDim)
	}
	for i := 0; i < r; i++ {
		ridx := i * C.stride
		for j := 0; j < c; j++ {
			C.data[ridx+j] = f * A.At(i, j)
		}
	}
}

// SwapRows swaps rows i and j of A in-place.
func (A DenseM) SwapRows(i, j int) {
	iidx := i * A.stride
	jidx := j * A.stride
	for k := 0; k < A.c; k++ {
		A.data[iidx+k], A.data[jidx+k] = A.data[jidx+k], A.data[iidx+k]
	}
}

func (A DenseM) SwapCols(i, j int) {
	for k := 0; k < A.r; k++ {
		ridx := k * A.stride
		A.data[ridx+i], A.data[ridx+j] = A.data[ridx+j], A.data[ridx+i]
	}
}

func (A DenseM) RowView(i int) DenseV {
	if i >= A.r || i < 0 {
		panic(ErrRowAccess)
	}
	return DenseV{
		data: A.data[i*A.stride : (i+1)*A.stride],
	}
}

func (A DenseM) ColView(j int) DenseV {
	if j >= A.c || j < 0 {
		panic(ErrColAccess)
	}
	return DenseV{
		data:        A.data[j:],
		incMinusOne: A.stride - 1,
	}
}

// CopyBlocks copies mrows rows and mcols columns of matrices
// passed in src.
func (dst *DenseM) CopyBlocks(mrows, mcols int, src []Matrix) error {
	if len(src) != mrows*mcols {
		return ErrDim
	}
	var tr, tc int
	for i := 0; i < mrows; i++ {
		r, _ := src[i*mcols].Dims()
		tr += r
	}
	for j := 0; j < mcols; j++ {
		_, c := src[j].Dims()
		tc += c
	}
	if dst.data == nil {
		*dst = NewDenseMatrix(tr, tc, nil)
	}
	r, c := dst.Dims()
	if r != tr || c != tc {
		return ErrDim
	}

	var br int
	for i := 0; i < mrows; i++ {
		var bc int
		h, _ := src[i*mcols].Dims()
		for j := 0; j < mcols; j++ {
			r, c := src[i*mcols+j].Dims()
			if r != h {
				return fmt.Errorf("matrix at %d,%d is wrong height: %d != %d:  %w", i, j, r, h, ErrDim)
			}
			if i != 0 {
				_, w := src[j].Dims()
				if c != w {
					return fmt.Errorf("matrix at %d,%d is wrong width: %d != %d:  %w", i, j, c, w, ErrDim)
				}
			}
			sli := dst.Slice(br, br+r, bc, bc+c)
			sli.Copy(src[i*mcols+j])
			bc += c
		}
		br += h
	}
	return nil
}

// DoSet iterates over all matrix elements calling fn on them and setting
// the value at i,j to the result of fn.
func (A DenseM) DoSet(fn func(i, j int, v float64) float64) {
	for i := 0; i < A.r; i++ {
		offset := i * A.stride
		for j := 0; j < A.c; j++ {
			got := A.data[offset+j]
			A.data[offset+j] = fn(i, j, got)
		}
	}
}
