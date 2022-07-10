package lap

var _ Matrix = subMat{}

// subMat allows
type subMat struct {
	cidx, ridx []int
	m          Matrix
}

func (bm subMat) At(i, j int) float64 { return bm.m.At(bm.ridx[i], bm.cidx[j]) }
func (bm subMat) Dims() (int, int)    { return len(bm.ridx), len(bm.cidx) }

// Slice slices a matrix given row and column indices.
func Slice(m Matrix, rowIx, colIx []int) Matrix {
	if len(rowIx) == 0 || len(colIx) == 0 {
		panic("cannot have zero dimension SubIdx")
	}
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

func (bm subVec) AtVec(i int) float64 { return bm.m.At(bm.ridx[i], 0) }
func (bm subVec) Len() int            { return len(bm.ridx) }

func SliceVec(v Vector, ix []int) Vector {
	sm := Slice(v, ix, []int{0}).(subMat)
	return subVec{subMat: sm}
}
