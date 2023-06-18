package lap

var _ Matrix = SliceM{}

// SliceM allows
type SliceM struct {
	cidx, ridx []int
	m          Matrix
}

func (sm SliceM) At(i, j int) float64 {
	lr := len(sm.ridx)
	lc := len(sm.cidx)
	if lr != 0 {
		i = sm.ridx[i]
	}
	if lc != 0 {
		j = sm.cidx[j]
	}
	return sm.m.At(i, j)
}

func (sm SliceM) Dims() (int, int) {
	r, c := sm.m.Dims()
	lr := len(sm.ridx)
	lc := len(sm.cidx)
	if lr != 0 {
		r = lr
	}
	if lc != 0 {
		c = lc
	}
	return r, c
}

// Slice slices a matrix given row and column indices.
// An empty slice means the entire row/column space is included.
func Slice(m Matrix, rowIx, colIx []int) SliceM {
	r, c := m.Dims()
	for _, ir := range rowIx {
		if ir >= r {
			panic(ErrRowAccess)
		}
	}
	for _, ic := range colIx {
		if ic >= c {
			panic(ErrColAccess)
		}
	}
	return SliceM{ridx: rowIx, cidx: colIx, m: m}
}

// IsModifiable returns true if the underlying matrix can be modified via Set and Copy calls.
func (sm SliceM) IsModifiable() bool {
	_, ok := sm.m.(matrixSetter)
	return ok
}

func (sm SliceM) Set(i, j int, v float64) {
	M, ok := sm.m.(matrixSetter)
	if !ok {
		panic("Set not implemented for underlying matrix")
	}
	lr := len(sm.ridx)
	lc := len(sm.cidx)
	if lr != 0 {
		i = sm.ridx[i]
	}
	if lc != 0 {
		j = sm.cidx[j]
	}
	M.Set(i, j, v)
}

// Copy copies the contents of the matrix into the provided matrix.
// The dimensions of the provided matrix must match the dimensions of the receiver.
func (sm SliceM) Copy(m Matrix) (rowsCopied, colsCopies int) {
	M, ok := sm.m.(matrixSetter)
	if !ok {
		panic("Set not implemented for underlying matrix")
	}
	gotr, gotc := m.Dims()
	r, c := sm.Dims()
	if r != gotr || c != gotc {
		panic(ErrDim)
	}
	for i := 0; i < r; i++ {
		ii := i
		if len(sm.ridx) != 0 {
			ii = sm.ridx[i]
		}
		for j := 0; j < c; j++ {
			jj := j
			if len(sm.cidx) != 0 {
				jj = sm.cidx[j]
			}
			M.Set(ii, jj, m.At(i, j))
		}
	}
	return r, c
}

type SliceV struct {
	sm SliceM
}

func (sv SliceV) AtVec(i int) float64 { return sv.sm.m.At(sv.sm.ridx[i], 0) }
func (sv SliceV) Len() int            { return len(sv.sm.ridx) }
func (sv SliceV) Dims() (int, int)    { return sv.sm.Dims() }
func (sv SliceV) At(i, j int) float64 { return sv.sm.At(i, j) }

func (sv SliceV) IsModifiable() bool {
	return sv.sm.IsModifiable()
}

func (sv SliceV) CopyVec(v Vector) int {
	r, _ := sv.sm.Copy(v)
	return r
}

func (sv SliceV) SetVec(i int, v float64) {
	sv.sm.Set(i, 0, v)
}

// SliceVec slices a vector given row indices. A zero length slice means the
// entire vector is included.
func SliceVec(v Vector, ix []int) SliceV {
	r, c := v.Dims()
	if c != 1 || r != v.Len() {
		panic("cannot slice a non-column vector")
	}
	sm := Slice(v, ix, []int{0})
	return SliceV{sm: sm}
}

// SliceExclude slices a matrix such that the provided indices are excluded
// from the resulting matrix. This is the counterpart to the Slice function.
func SliceExclude(m Matrix, excludeRows, excludeCols []int) SliceM {
	r, c := m.Dims()
	lastRow := -1
	for _, ir := range excludeRows {
		if ir >= r {
			panic(ErrRowAccess)
		}
		if ir <= lastRow {
			panic("row indices must be sorted and non-repeating")
		}
		lastRow = ir
	}
	lastCol := -1
	for _, ic := range excludeCols {
		if ic >= c {
			panic(ErrColAccess)
		}
		if ic <= lastCol {
			panic("column indices must be sorted and non-repeating")
		}
		lastCol = ic
	}
	rowIx := make([]int, r-len(excludeRows))
	colIx := make([]int, c-len(excludeCols))
	nextRow := 0
	for i := 0; i < r; i++ {
		if nextRow < len(excludeRows) && i == excludeRows[nextRow] {
			nextRow++
			continue
		}
		rowIx[i-nextRow] = i
	}
	nextCol := 0
	for i := 0; i < c; i++ {
		if nextCol < len(excludeCols) && i == excludeCols[nextCol] {
			nextCol++
			continue
		}
		colIx[i-nextCol] = i
	}
	return SliceM{
		cidx: colIx,
		ridx: rowIx,
		m:    m,
	}
}

// SliceExcludeVec slices a vector such that the provided indices are excluded.
func SliceExcludeVec(v Vector, excludeIdx []int) SliceV {
	sm := SliceExclude(v, excludeIdx, nil)
	return SliceV{sm: sm}
}
