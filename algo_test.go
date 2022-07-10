package lap

import "testing"

func TestJacobiSVD(t *testing.T) {
	var m DenseM
	m.Copy(magic3)
	expect := NewDenseVector(3, []float64{3.4641, 6.9282, 15})
	sigma := JacobiSVD(m)
	if !vectorEqualTol(sigma, expect, 0.01) {
		t.Error("sigma not equal to expect, ", sigma, expect)
	}
}
