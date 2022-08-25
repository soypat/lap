package lap

// SparseAccum is a helper struct for constructing sparse matrices efficiently.
type SparseAccum struct {
	// Row indices
	I []int
	// Column indices
	J []int
	// Matrix values.
	V []float64
}

// SparseAccum creates a sparse matrix element accumulator of length n.
func NewSparseAccum(n int) SparseAccum {
	if n < 1 {
		panic("amount of elements in SparseAccum must be greater than 0")
	}
	return SparseAccum{
		I: make([]int, n),
		J: make([]int, n),
		V: make([]float64, n),
	}
}

// Set sets the the element of the supposed matrix at position i, j to v.
// offset is the unique position inside the accumulator.
func (sp SparseAccum) Set(offset, i, j int, v float64) {
	sp.I[offset] = i
	sp.J[offset] = j
	sp.V[offset] = v
}

// Zero sets all values and indices of the accumulator to zero.
func (sp SparseAccum) Zero() {
	if len(sp.I) != len(sp.J) || len(sp.V) != len(sp.I) {
		panic("lengths should be equal")
	}
	for i := range sp.I {
		sp.I[i] = 0
		sp.J[i] = 0
		sp.V[i] = 0
	}
}

// Sparse is experimental sparse matrix.
type Sparse struct {
	m    map[[2]int]float64
	r, c int
}

var _ Matrix = Sparse{}

// At returns the element at ith row, jth column.
func (s Sparse) At(i, j int) float64 {
	if i >= s.r {
		panic(ErrRowAccess)
	} else if j >= s.c {
		panic(ErrColAccess)
	}
	return s.m[[2]int{i, j}]
}

// At sets the element at ith row, jth column to v.
func (s Sparse) Set(i, j int, v float64) {
	if i >= s.r {
		panic(ErrRowAccess)
	} else if j >= s.c {
		panic(ErrColAccess)
	} else if s.m == nil {
		s.m = make(map[[2]int]float64)
	}
	ix := [2]int{i, j}
	if v != 0 {
		s.m[ix] = v
	} else {
		delete(s.m, ix)
	}
}

// Dims returns the dimensions of the sparse matrix.
func (s Sparse) Dims() (int, int) {
	return s.r, s.c
}

// NewSparse creates a sparse matrix of size rxc.
func NewSparse(r, c int) Sparse {
	s := Sparse{
		r: r,
		c: c,
		m: make(map[[2]int]float64),
	}
	return s
}

// Resize gives sparse matrix a new size and discards old data out of bounds
// (right and lower matrix).
func (s Sparse) Resize(r, c int) {
	if s.m == nil {
		s.m = make(map[[2]int]float64)
	}
	if r < s.r || c < s.c {
		for k := range s.m {
			if k[0] >= r || k[1] >= c {
				delete(s.m, k)
			}
		}
	}
	s.r = r
	s.c = c
}

// Accumulate adds indexed data to s.
func (s Sparse) Accumulate(data SparseAccum) {
	s.GeneralAccumulate(false, 0, 0, data)
}

// GeneralAccumulate adds indexed data to s. The input data can be transposed
// with trans set to true. s index offsets may be set with iOffset and jOffset.
func (s Sparse) GeneralAccumulate(trans bool, iOffset, jOffset int, data SparseAccum) {
	if len(data.I) != len(data.J) || len(data.V) != len(data.I) {
		panic("length of arguments must be equal")
	}
	if len(s.m) == 0 {
		s.m = make(map[[2]int]float64, len(data.V)/16)
	}
	var ix, jx int
	for i, v := range data.V {

		if trans {
			jx, ix = data.I[i], data.J[i]
		} else {
			ix, jx = data.I[i], data.J[i]
		}

		if ix >= s.r {
			panic(ErrRowAccess)
		} else if jx >= s.c {
			panic(ErrColAccess)
		}
		if v == 0 {
			continue // This gives a considerable speed boost as of 1.18.
		}
		idx := [2]int{ix + iOffset, jx + jOffset}
		s.m[idx] += v
	}
}

// DoNonZero iterates over all non-zero values in sparse matrix.
// The function fn takes a row/column index and the element value of b at (i, j).
// No specific ordering is guaranteed.
func (s Sparse) DoNonZero(fn func(i, j int, v float64)) {
	for k, v := range s.m {
		fn(k[0], k[1], v)
	}
}

// CountNonZero returns number of non-zero elements in sparse matrix.
func (s Sparse) CountNonZero() int { return len(s.m) }

// Zero sets all matrix values to zero.
func (s Sparse) Zero() {
	s.m = nil
}
