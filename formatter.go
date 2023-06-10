// This file taken from gonum/mat/formatter.go
//
// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lap

import (
	"fmt"
	"strconv"
	"strings"
)

// Formatted returns a fmt.Formatter for the matrix m using the given options.
// It does not support FormatOptions yet.
func Formatted(m Matrix, options ...FormatOption) fmt.Formatter {
	f := formatter{
		matrix: m,
		dot:    '.',
	}
	for _, o := range options {
		o(&f)
	}
	return f
}

type formatter struct {
	matrix  Matrix
	prefix  string
	margin  int
	dot     byte
	squeeze bool

	format func(m Matrix, prefix string, margin int, dot byte, squeeze bool, fs fmt.State, c rune)
}

// FormatOption is a functional option for matrix formatting.
type FormatOption func(*formatter)

// Format satisfies the fmt.Formatter interface.
func (f formatter) Format(fs fmt.State, c rune) {
	if c == 'v' && fs.Flag('#') && f.format == nil {
		fmt.Fprintf(fs, "%#v", f.matrix)
		return
	}
	if f.format == nil {
		f.format = formatMATLAB
	}
	f.format(f.matrix, f.prefix, f.margin, f.dot, f.squeeze, fs, c)
}

// formatMATLAB prints a MATLAB representation of m to the fs io.Writer. The format character c
// specifies the numerical representation of elements; valid values are those for float64
// specified in the fmt package, with their associated flags.
// The printed range of the matrix can be limited by specifying a positive value for margin;
// If squeeze is true, column widths are determined on a per-column basis.
//
// formatMATLAB will not provide Go syntax output.
func formatMATLAB(m Matrix, prefix string, _ int, _ byte, squeeze bool, fs fmt.State, c rune) {
	rows, cols := m.Dims()

	prec, pOk := fs.Precision()
	width, _ := fs.Width()
	if !fs.Flag('#') {
		switch c {
		case 'v', 'e', 'E', 'f', 'F', 'g', 'G':
		default:
			fmt.Fprintf(fs, "%%!%c(%T=Dims(%d, %d))", c, m, rows, cols)
			return
		}
		format := fmtString(fs, c, prec, width)
		fs.Write([]byte{'['})
		for i := 0; i < rows; i++ {
			if i != 0 {
				fs.Write([]byte("; "))
			}
			for j := 0; j < cols; j++ {
				if j != 0 {
					fs.Write([]byte{' '})
				}
				fmt.Fprintf(fs, format, m.At(i, j))
			}
		}
		fs.Write([]byte{']'})
		return
	}

	if !pOk {
		prec = -1
	}

	printed := rows
	if cols > printed {
		printed = cols
	}

	var (
		maxWidth int
		widths   widther
		buf, pad []byte
	)
	if squeeze {
		widths = make(columnWidth, cols)
	} else {
		widths = new(uniformWidth)
	}
	switch c {
	case 'v', 'e', 'E', 'f', 'F', 'g', 'G':
		if c == 'v' {
			buf, maxWidth = maxCellWidth(m, 'g', printed, prec, widths)
		} else {
			buf, maxWidth = maxCellWidth(m, c, printed, prec, widths)
		}
	default:
		fmt.Fprintf(fs, "%%!%c(%T=Dims(%d, %d))", c, m, rows, cols)
		return
	}
	width = max(width, maxWidth)
	pad = make([]byte, max(width, 1))
	for i := range pad {
		pad[i] = ' '
	}

	for i := 0; i < rows; i++ {
		var el string
		switch {
		case rows == 1:
			fmt.Fprint(fs, "[")
			el = "]"
		case i == 0:
			fmt.Fprint(fs, "[\n"+prefix+" ")
			el = "\n"
		case i < rows-1:
			fmt.Fprint(fs, prefix+" ")
			el = "\n"
		default:
			fmt.Fprint(fs, prefix+" ")
			el = "\n" + prefix + "]"
		}

		for j := 0; j < cols; j++ {
			v := m.At(i, j)
			if c == 'v' {
				buf = strconv.AppendFloat(buf[:0], v, 'g', prec, 64)
			} else {
				buf = strconv.AppendFloat(buf[:0], v, byte(c), prec, 64)
			}
			if fs.Flag('-') {
				fs.Write(buf)
				fs.Write(pad[:widths.width(j)-len(buf)])
			} else {
				fs.Write(pad[:widths.width(j)-len(buf)])
				fs.Write(buf)
			}

			if j < cols-1 {
				fs.Write(pad[:1])
			}
		}

		fmt.Fprint(fs, el)
	}
}

// This is horrible, but it's what we have.
func fmtString(fs fmt.State, c rune, prec, width int) string {
	var b strings.Builder
	b.WriteByte('%')
	for _, f := range "0+- " {
		if fs.Flag(int(f)) {
			b.WriteByte(byte(f))
		}
	}
	if width >= 0 {
		fmt.Fprint(&b, width)
	}
	if prec >= 0 {
		b.WriteByte('.')
		if prec > 0 {
			fmt.Fprint(&b, prec)
		}
	}
	b.WriteRune(c)
	return b.String()
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

type widther interface {
	width(i int) int
	setWidth(i, w int)
}

type uniformWidth int

func (u *uniformWidth) width(_ int) int   { return int(*u) }
func (u *uniformWidth) setWidth(_, w int) { *u = uniformWidth(w) }

type columnWidth []int

func (c columnWidth) width(i int) int   { return c[i] }
func (c columnWidth) setWidth(i, w int) { c[i] = w }

func maxCellWidth(m Matrix, c rune, printed, prec int, w widther) ([]byte, int) {
	var (
		buf        = make([]byte, 0, 64)
		rows, cols = m.Dims()
		max        int
	)
	for i := 0; i < rows; i++ {
		if i >= printed-1 && i < rows-printed && 2*printed < rows {
			i = rows - printed - 1
			continue
		}
		for j := 0; j < cols; j++ {
			if j >= printed && j < cols-printed {
				continue
			}

			buf = strconv.AppendFloat(buf, m.At(i, j), byte(c), prec, 64)
			if len(buf) > max {
				max = len(buf)
			}
			if len(buf) > w.width(j) {
				w.setWidth(j, len(buf))
			}
			buf = buf[:0]
		}
	}
	return buf, max
}
