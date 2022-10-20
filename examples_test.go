package lap_test

import (
	"math"
	"math/rand"

	"github.com/soypat/lap"
)

func ExampleDenseV_neuralNetwork() {
	// Inspired directly by Santiago's tweet
	// https://gist.github.com/svpino/e54ff030c424cefaffeec1bd690042cc  https://twitter.com/svpino/status/1582703127651721217?t=g3aXpwKbqCBIW9AJYzgKgA&s=08
	const (
		learningRate = 0.1
		inputSize    = 4
		hiddenSize   = 2
		outputSize   = 1
		epochs       = 10000
	)
	rng := rand.New(rand.NewSource(1))
	randomMatrix := func(n, m int) lap.DenseM {
		size := n * m
		v := make([]float64, size)
		for i := 0; i < size; i++ {
			v[i] = rng.Float64()
		}
		return lap.NewDenseMatrix(n, m, v)
	}
	// Sigmoid activation function.
	applySigmoid := func(x lap.DenseM) {
		x.DoSet(func(_, _ int, v float64) float64 { return 1.0 / (1 + math.Exp(-v)) })
	}

	// weights that connect the input with layer1
	W1 := randomMatrix(hiddenSize, inputSize)
	// weights that connect layer1 with output.
	W2 := randomMatrix(inputSize, outputSize)
	// Inputs are X and y.
	X := lap.NewDenseMatrix(inputSize, hiddenSize, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	y := lap.NewDenseVector(inputSize, []float64{0, 1, 1, 1})

	var layer1, output, aux, aux2, nnerror, delta2 lap.DenseM
	for epoch := 0; epoch < epochs; epoch++ {
		layer1.Mul(X, W1)
		applySigmoid(layer1)
		output.Mul(layer1, W2)
		applySigmoid(output)
		nnerror.Sub(y, output)

		aux.Copy(output)
		aux.DoSet(func(_, _ int, v float64) float64 { return 1 - v })
		// delta2.Mul(output, aux)
		// aux2.Mul(nnerror, delta2)
		// Example under construction.
	}
	_, _ = aux2, delta2
	// fmt.Println(output)
	//Output:

}
