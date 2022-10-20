package lap32

import (
	"fmt"
	"reflect"
	"unsafe"
)

type Vector interface {
	Matrix
	AtVec(i int) float32
	Len() int
}

type DenseV struct {
	data        []float32
	incMinusOne int
}

// NewDenseVector returns a vector of length n with data. If data is nil it is
// automatically allocated.
func NewDenseVector(n int, data []float32) DenseV {
	if data == nil {
		data = make([]float32, n)
	}
	if len(data) != n {
		panic(ErrDim)
	}
	return DenseV{
		data: data,
	}
}

func (v DenseV) Dims() (int, int) { return v.Len(), 1 }
func (v DenseV) At(i, j int) float32 {
	if j != 0 {
		panic(ErrColAccess)
	}
	return v.AtVec(i)
}

func (v DenseV) Len() int {
	l := len(v.data)
	if v.incMinusOne != 0 {
		div, mod := l/(v.incMinusOne+1), l%(v.incMinusOne+1)
		if mod == 0 {
			return div
		} else {
			return div + 1
		}
	}
	return l
}

func (v DenseV) AtVec(i int) float32 {
	if v.incMinusOne != 0 {
		return v.data[i*(v.incMinusOne+1)]
	}
	return v.data[i]
}

func (v *DenseV) SetVec(i int, f float32) {
	if v.incMinusOne != 0 {
		v.data[i*(v.incMinusOne+1)] = f
	} else {
		v.data[i] = f
	}
}

// AddVec adds the vectors a+b element-wise, placing the result in the receiver.
func (v *DenseV) AddVec(a, b Vector) {
	n := v.Len()
	if v.data == nil {
		*v = NewDenseVector(n, nil)
	}
	if n != b.Len() || n != a.Len() {
		panic(ErrDim)
	}
	for i := 0; i < n; i++ {
		v.SetVec(i, a.AtVec(i)+b.AtVec(i))
	}
}

// SubVec subtracts the vectors a-b element-wise, placing the result in the receiver.
func (v *DenseV) SubVec(a, b Vector) {
	n := v.Len()
	if v.data == nil {
		*v = NewDenseVector(n, nil)
	}
	if n != b.Len() || n != a.Len() {
		panic(ErrDim)
	}
	for i := 0; i < n; i++ {
		v.SetVec(i, a.AtVec(i)-b.AtVec(i))
	}
}

// CopyVec makes a copy of elements of a into the receiver and returns the amount
// of elements copied. If the receiver has not been initialized then a vector is allocated.
func (v *DenseV) CopyVec(a Vector) int {
	n := a.Len()
	if v.data == nil {
		*v = NewDenseVector(n, nil)
	}
	if n != v.Len() {
		panic(ErrDim)
	}
	for i := 0; i < n; i++ {
		v.SetVec(i, a.AtVec(i))
	}
	return n
}

// MulVec computes A * b. The result is stored into the receiver.
// MulVec panics if the number of columns in A does not equal the number of
// rows in b or if the number of columns in b does not equal 1.
func (v *DenseV) MulVec(A Matrix, b Vector) {
	n := b.Len()
	m, c := A.Dims()
	if c != n {
		panic(ErrDim)
	}
	if v.data == nil {
		*v = NewDenseVector(m, nil)
	} else if aliasedData(v, b) || aliasedData(v, A) {
		panic(ErrAliasedData)
	}
	if m != v.Len() {
		panic(ErrDim)
	}
	for i := 0; i < m; i++ {
		var tmp float32
		for j := 0; j < n; j++ {
			tmp += A.At(i, j) * b.AtVec(j)
		}
		v.SetVec(i, tmp)
	}
}

func (v *DenseV) MulElemVec(a, b Vector) {
	ar := a.Len()
	if v.data == nil {
		*v = NewDenseVector(ar, nil)
	}
	br := b.Len()
	if ar != br {
		panic(ErrDim)
	}
	if v.Len() != ar {
		panic(ErrDim)
	}
	for i := 0; i < ar; i++ {
		v.SetVec(i, a.AtVec(i)*b.AtVec(i))
	}
}

// DoSet iterates over all vector elements calling fn on them and setting
// the value at i to the result of fn.
func (A DenseV) DoSetVec(fn func(i int, v float32) float32) {
	// TODO(soypat): This could be optimized for direct access.
	n := A.Len()
	for i := 0; i < n; i++ {
		A.SetVec(i, fn(i, A.AtVec(i)))
	}
}

func aliasedData(a, b Matrix) bool {
	dataA := dataHeader(a)
	dataB := dataHeader(b)
	if dataA.Data > dataB.Data {
		// Sort in order of increasing data position in memory.
		dataA, dataB = dataB, dataA
	}
	return dataA.Len > 0 && dataB.Len > 0 && dataA.Data+uintptr(dataA.Len) > dataB.Data
}

func dataHeader(m Matrix) reflect.SliceHeader {
	var backingData []float32
	switch D := m.(type) {
	case *DenseM:
		backingData = D.data
	case DenseM:
		backingData = D.data
	case DenseV:
		backingData = D.data
	case subMat:
		return dataHeader(D.m)
	case subVec:
		return dataHeader(D.m)
	case Transpose:
		return dataHeader(D.m)
	default:
		panic("unknown Matrix type. Can't determine backing data" + fmt.Sprintf("%T", D))
	}
	return *(*reflect.SliceHeader)(unsafe.Pointer(&backingData))
}
