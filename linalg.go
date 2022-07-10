// package lap is a pure-Go implementation of compact linear algebra operations.
//
// It has the purpose of being a smaller dependency than gonum for use
// in constrained applications, such as deploying to web and embedded systems.
//
// It shall always be buildable with tinygo.

package lap

import (
	"math"
)

// Dot returns the sum of the element-wise product of a and b.
//
// Dot panics with ErrShape if the vector sizes are unequal.
func Dot(a, b Vector) float64 {
	la := a.Len()
	lb := b.Len()
	if la != lb {
		panic(ErrDim)
	}
	if la == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < la; i++ {
		sum += a.At(i, 0) * b.At(i, 0)
	}
	return sum
}

// Max returns the largest element value of the matrix A.
func Max(m Matrix) float64 {
	r, c := m.Dims()
	max := math.Inf(-1)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			if v > max {
				max = v
			}
		}
	}
	return max
}

// Min returns the smallest element value of the matrix A.
func Min(m Matrix) float64 {
	r, c := m.Dims()
	min := math.Inf(1)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			if v > min {
				min = v
			}
		}
	}
	return min
}

// Norm returns the specified norm of the matrix A. Valid norms are:
//  1 - The maximum absolute column sum
//  2 - The Frobenius norm, the square root of the sum of the squares of the elements
//  Inf - The maximum absolute row sum
func Norm(A Matrix, norm float64) float64 {
	r, c := A.Dims()
	switch norm {
	default:
		panic("Bad norm order, accept 1, 2, +Inf")
	case 1:
		var max float64
		for j := 0; j < c; j++ {
			var sum float64
			for i := 0; i < r; i++ {
				sum += math.Abs(A.At(i, j))
			}
			if sum > max {
				max = sum
			}
		}
		return max
	case 2:
		var sum float64
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				v := A.At(i, j)
				sum += v * v
			}
		}
		return math.Sqrt(sum)
	case math.Inf(1):
		var max float64
		for i := 0; i < r; i++ {
			var sum float64
			for j := 0; j < c; j++ {
				sum += math.Abs(A.At(i, j))
			}
			if sum > max {
				max = sum
			}
		}
		return max
	}
}

// Sum returns the sum of the elements of the matrix.
func Sum(A Matrix) (sum float64) {
	r, c := A.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum += A.At(i, j)
		}
	}
	return sum
}

// Argmax returns indices of maximum value in matrix.
func Argmax(A Matrix) (imax, jmax int) {
	max := math.Inf(-1)
	r, c := A.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := A.At(i, j)
			if v > max {
				max = v
				imax = i
				jmax = j
			}
		}
	}
	return imax, jmax
}
