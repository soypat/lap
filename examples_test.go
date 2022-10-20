package lap_test

import (
	"math"
	"math/rand"

	"github.com/soypat/lap"
)

func ExampleDenseM_neuralNetwork() {
	// Inspired directly by Santiago's tweet
	// https://gist.github.com/svpino/e54ff030c424cefaffeec1bd690042cc  https://twitter.com/svpino/status/1582703127651721217?t=g3aXpwKbqCBIW9AJYzgKgA&s=08
	const (
		learningRate = 0.1
		casesSize    = 4
		inputSize    = 2
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
	W1 := randomMatrix(inputSize, casesSize)
	// weights that connect layer1 with output.
	W2 := randomMatrix(casesSize, outputSize)
	// cases represents all possible inputs.
	// Each row is a input to the neural network.
	cases := lap.NewDenseMatrix(casesSize, inputSize, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	ORresult := lap.NewDenseVector(casesSize, []float64{0, 1, 1, 1})

	var layer1, output, aux1, aux2, aux3, aux4, aux5, nnerror, delta1, delta2 lap.DenseM
	for epoch := 0; epoch < epochs; epoch++ {
		// Example under construction.
		layer1.Mul(cases, W1)
		applySigmoid(layer1)
		output.Mul(layer1, W2)
		applySigmoid(output)
		nnerror.Sub(ORresult, output)

		aux1.Copy(output)
		aux1.DoSet(func(_, _ int, v float64) float64 { return 1 - v })
		d2 := lap.Dot(output.ColView(0), aux1.ColView(0))
		delta2.Scale(2*d2, nnerror)

		aux2.Copy(layer1)
		aux2.DoSet(func(i, j int, v float64) float64 { return 1 - v })
		aux3.Mul(layer1, aux2)
		aux4.Mul(delta2, lap.T(W2))
		delta1.Mul(aux4, aux3)
		// Prepare modifying neural network nodes.
		aux1.Mul(lap.T(layer1), delta2)
		aux1.Scale(learningRate, aux1)
		W2.Add(W2, aux1)

		aux5.Mul(lap.T(cases), delta1)
		aux5.Scale(learningRate, aux5)
		W1.Add(W1, aux5)
	}
	// fmt.Println(output)
	// Should yield
	//{[0 1 1 1] 1 4 1}
	//Output:
}
