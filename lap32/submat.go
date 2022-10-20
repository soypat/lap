package lap32

var _ Matrix = subMat{}

// subMat allows
type subMat struct {
	cidx, ridx []int
	m          Matrix
}

func (bm subMat) At(i, j int) float32 {
	lr := len(bm.ridx)
	lc := len(bm.cidx)
	if lr != 0 {
		i = bm.ridx[i]
	}
	if lc != 0 {
		j = bm.cidx[j]
	}
	return bm.m.At(i, j)
}

func (bm subMat) Dims() (int, int) {
	r, c := bm.m.Dims()
	lr := len(bm.ridx)
	lc := len(bm.cidx)
	if lr != 0 {
		r = lr
	}
	if lc != 0 {
		c = lc
	}
	return r, c
}

// Slice slices a matrix given row and column indices.
// an empty slice means the entire dimension is taken.
func Slice(m Matrix, rowIx, colIx []int) Matrix {
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
	return subMat{ridx: rowIx, cidx: colIx, m: m}
}

type subVec struct {
	subMat
}

func (bm subVec) AtVec(i int) float32 { return bm.m.At(bm.ridx[i], 0) }
func (bm subVec) Len() int            { return len(bm.ridx) }

func SliceVec(v Vector, ix []int) Vector {
	sm := Slice(v, ix, []int{0}).(subMat)
	return subVec{subMat: sm}
}
