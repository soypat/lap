package lap

import (
	"math"
	"reflect"
	"testing"
	"unsafe"
)

const almostEps = 1e-16

func TestNewDenseMatrix(t *testing.T) {
	n := 4
	m := 6
	// will panic if new dense accesses invalid memory
	mat := NewDenseMatrix(n, m, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			v := mat.At(i, j)
			if v != 0 {
				t.Error("matrix was expected to be zero")
			}
		}
	}
}

func TestSwapRows_Copy(t *testing.T) {
	A := NewDenseMatrix(2, 3, []float64{
		0, 1, 2,
		3, 4, 5,
	})
	var B DenseM
	B.CopyFrom(A)
	B.SwapRows(0, 1)
	for i := 0; i < 3; i++ {
		at := B.At(0, i)
		if at != float64(i+3) {
			t.Errorf("row 0 at position %d had value %.0f, expected %d", i, at, i+3)
		}
	}
}

func TestDims(t *testing.T) {
	A := NewDenseMatrix(3, 4, nil)
	m, n := A.Dims()
	if m != 3 || n != 4 {
		t.Errorf("matrix had shape (%d,%d), expected (3,4)", m, n)
	}
}

func TestEye(t *testing.T) {
	A := Eye(3)
	var expectation float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if i == j {
				expectation = 1
			} else {
				expectation = 0
			}
			at := A.At(i, j)
			if at != expectation {
				t.Errorf("at r,c (%d,%d) identity matrix had value %f, expected %f", i, j, at, expectation)
			}
		}
	}
}

func TestMatCopy(t *testing.T) {
	// MatCopy produces a copy of A with no overlapping memory, this tests that
	// the pointers truly do not overlap.  TestMatCopyTo verifies that the
	// elements are the same.  This is fragile, in that we assume they share
	// an implementation.  _shrug_
	A := NewDenseMatrix(4, 4, nil)
	var B DenseM
	B.CopyFrom(A)
	dataptrA := (*reflect.SliceHeader)(unsafe.Pointer(&A.data)).Data
	dataptrB := (*reflect.SliceHeader)(unsafe.Pointer(&B.data)).Data
	if dataptrA == dataptrB {
		t.Error("had same data pointer, expected to be different")
	}
}

func TestMatCopyTo(t *testing.T) {
	A := NewDenseMatrix(4, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16})

	B := NewDenseMatrix(4, 4, nil)
	B.CopyFrom(A)
	m, n := A.Dims()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a := A.At(i, j)
			b := B.At(i, j)
			if a != b {
				t.Errorf("at (%d,%d), expected A==B, got %f and %f", i, j, a, b)
			}
		}
	}
}

func TestMatL2Norm(t *testing.T) {
	data := []float64{
		1, 2, 3,
		4, 5, 6}
	A := NewDenseMatrix(2, 3, data)
	var expectation float64
	for _, v := range data {
		expectation += v * v
	}
	expectation = math.Sqrt(expectation)
	norm := A.Norm(2)
	if !almostEqual(expectation, norm, almostEps) {
		t.Errorf("expected %f, got %f", expectation, norm)
	}
}

func TestMatTranspose(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6})
	exp := NewDenseMatrix(3, 2, []float64{
		1, 4,
		2, 5,
		3, 6})
	inpT := T(inp)
	if !matrixEqual(exp, inpT) {
		t.Error("matrix transpose did not match expectation")
	}
}

func TestMatAdd(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	sub := NewDenseMatrix(2, 3, []float64{
		2, 3, 4,
		5, 6, 7,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		3, 5, 7,
		9, 11, 13,
	})
	out := NewDenseMatrix(2, 3, nil)
	out.Add(inp, sub)
	if !matrixEqual(exp, out) {
		t.Error("matrix sub did not match expectation")
	}
}

func TestMatSub(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	sub := NewDenseMatrix(2, 3, []float64{
		2, 3, 4,
		5, 6, 7,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		-1, -1, -1,
		-1, -1, -1,
	})
	out := NewDenseMatrix(2, 3, nil)
	out.Sub(inp, sub)
	if !matrixEqual(exp, out) {
		t.Error("matrix sub did not match expectation")
		t.Log(out)
	}
}

func TestMatSwapRows(t *testing.T) {
	inp1 := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	inp2 := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		4, 5, 6,
		1, 2, 3,
	})
	inp1.SwapRows(0, 1)
	inp2.SwapRows(0, 1)
	if !matrixEqual(exp, inp1) {
		t.Error("matrix swap rows non copying did not match expectation")
	}
	if !matrixEqual(exp, inp2) {
		t.Error("matrix swap rows copying did not match expectation")
	}
}

func TestMatSwapCols(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	exp := NewDenseMatrix(2, 3, []float64{
		2, 1, 3,
		5, 4, 6,
	})
	inp.SwapCols(0, 1)
	if !matrixEqual(exp, inp) {
		t.Error("matrix swap cols did not match expectation")
	}
}

func TestMatColView(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	cols := [][]float64{
		{1, 4}, {2, 5}, {3, 6},
	}
	for j := range cols {
		got := inp.ColView(j)
		expect := NewDenseVector(got.Len(), cols[j])
		if !vectorEqual(got, expect) {
			t.Error("got column not equal")
		}
	}
}

func TestMatRowView(t *testing.T) {
	inp := NewDenseMatrix(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	rows := [][]float64{
		{1, 2, 3}, {4, 5, 6},
	}
	for i := range rows {
		got := inp.RowView(i)
		expect := NewDenseVector(got.Len(), rows[i])
		if !vectorEqual(got, expect) {
			t.Error("got row not equal", i)
		}
	}
}

// func TestSquareMatrixInvert(t *testing.T) {
// 	inp := NewDenseMatrix(2, 2, []float64{
// 		1, 2,
// 		3, 4,
// 	})
// 	exp := NewDenseMatrix(2, 2, []float64{
// 		-2, 1,
// 		1.5, -0.5,
// 	})
// 	var inv DenseM
// 	err := inv.invertSquare(inp, nil)
// 	if err != nil {
// 		t.Fatal(err)
// 	}
// 	if !matrixEqual(exp, inv) {
// 		t.Error("matrix inversion did not match expectation")
// 	}
// }
