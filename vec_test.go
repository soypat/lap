package lap

import (
	"testing"
)

// MatVecProd
// VectorAdd
// VectorSub
// VectorArgmax

func TestVectorAdd(t *testing.T) {
	a := NewDenseVector(5, []float64{1, 2, 3, 4, 5})
	b := NewDenseVector(5, []float64{5, 4, 3, 2, 1})
	exp := NewDenseVector(5, []float64{6, 6, 6, 6, 6})
	a.AddVec(a, b)
	if !vectorEqualTol(exp, a, 1e-16) {
		t.Error("vector addition failed to produce the expected result")
	}
}

func TestVectorSub(t *testing.T) {
	a := NewDenseVector(5, []float64{1, 2, 3, 4, 5})
	b := NewDenseVector(5, []float64{5, 4, 3, 2, 1})
	exp := NewDenseVector(5, []float64{-4, -2, 0, 2, 4})
	a.SubVec(a, b)
	if !vectorEqualTol(exp, a, 1e-16) {
		t.Error("vector addition failed to produce the expected result")
	}
}

func TestMatVecProd(t *testing.T) {
	A := NewDenseMatrix(2, 2, []float64{
		0, 1,
		2, 3,
	})
	b := NewDenseVector(2, []float64{9, 8})
	exp := NewDenseVector(2, []float64{8, 42})
	var res DenseV
	res.MulVec(A, b)
	if !vectorEqual(exp, res) {
		t.Error("matrix-vector product gave incorrect result", exp, res)
	}
}
