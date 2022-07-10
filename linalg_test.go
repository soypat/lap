package lap

import (
	"math"
	"testing"
)

var magic3 Matrix = NewDenseMatrix(3, 3, []float64{
	8, 1, 6,
	3, 5, 7,
	4, 9, 2})

func TestAliasedData(t *testing.T) {
	a := NewDenseVector(10, nil)
	b := NewDenseVector(20, nil)
	if aliasedData(a, b) {
		t.Fatal("new vectors should not be aliased")
	}
	b.data = a.data
	if !aliasedData(a, b) {
		t.Fatal("vectors should be aliased")
	}
}

func TestArgmax(t *testing.T) {
	mat := magic3
	i, j := Argmax(mat)
	if i != 2 || j != 1 {
		t.Errorf("bad argmax return value")
	}
	if mat.At(i, j) != Max(mat) {
		t.Errorf("bad max or argmax return value")
	}
}

func almostEqual(a, b, tol float64) bool {
	if tol == 0 {
		return a == b
	}
	return math.Abs(a-b) <= tol
}

func matrixEqualTol(A, B Matrix, tol float64) bool {
	m, n := A.Dims()
	mB, nB := B.Dims()
	if mB != m || nB != n {
		return false
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if !almostEqual(A.At(i, j), B.At(i, j), tol) {
				return false
			}
		}
	}
	return true
}

func matrixEqual(A, B Matrix) bool {
	return matrixEqualTol(A, B, 0)
}

func vectorEqualTol(a, b Vector, tol float64) bool {
	n := a.Len()
	if n != b.Len() {
		return false
	}
	for i := 0; i < n; i++ {
		if !almostEqual(a.AtVec(i), b.AtVec(i), tol) {
			return false
		}
	}
	return true
}

func vectorEqual(a, b Vector) bool {
	return vectorEqualTol(a, b, 0)
}
