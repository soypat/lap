package lap

import (
	"math/rand"
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

func TestVecSet(t *testing.T) {
	const maxLen = 10
	var scratch [maxLen]float64
	// rng := rand.New(rand.NewSource(1))
	Nnow := 3
	Nnext := 5
	for Nnext < maxLen {
		// Test all fibonacci numbers up to maxLen
		v := NewDenseVector(Nnow, scratch[:Nnow])
		N := v.Len()
		if N != Nnow {
			t.Fatal("calculated length not match set length")
		}
		for i := 0; i < N; i++ {
			fi := float64(i)
			v.SetVec(i, fi)
			if v.AtVec(i) != fi {
				t.Errorf("vector length %d got invalid set value at position %d", N, i)
			}
		}
		for i := 0; i < N; i++ {
			fi := float64(i)
			if v.AtVec(i) != fi {
				t.Errorf("vector length %d got invalid set value at position %d on final pass", N, i)
			}
		}
		for i := N - 1; i >= 0; i-- {
			fi := -float64(i)
			v.SetVec(i, fi)
			if v.AtVec(i) != fi {
				t.Errorf("vector length %d got invalid set value at position %d", N, i)
			}
			// Work way up to i
			for j := 0; j < i; j++ {
				if v.AtVec(j) != float64(j) {
					t.Errorf("vector length %d got invalid set value at position %d in countup", N, j)
				}
			}
		}
		scratch = [maxLen]float64{}
		scratch[0] = -2
		Nnow, Nnext = Nnext, Nnext+Nnow
	}
}

func randomSlice(rng *rand.Rand, n int) []float64 {
	v := make([]float64, n)
	for i := 0; i < n; i++ {
		v[i] = rng.Float64()
	}
	return v
}
