// Notes: Kevin did something like this and got it to learn simple I/O pairs

// Neurons:
// Excitatory conductance, inhib conduct, leak,
// those feed into membrane potential,
// spiking when membrane potential is over a threshold

// Try constant lateral inhibition

// Learning can be hebbian or STDP or whatever
// can do the kinase learning thing whatever

// Later: lateral inhib, NMDA, frequency regulation

package main

import (
	"math/rand"
	"strconv"
)

// Hyperparameters
var spikeThreshold float32 = 0.3
var leak float32 = 0.8
var weightMultiplier float32 = 0.1 // This exists solely so that weights can be numbers that I like
var weightChangeLearningRate float32 = 0.01
var deltaTime float32 = 1 // in milliseconds
var inhibitoryWeight float32 = 0.1
var targetActiveRatio float32 = 0.20          // This probably has a big effect
var regulatoryAdjustmentWeight float32 = 0.01 // Should be like 0.01
var excitatoryCloseDelay int = 5
var inhibitoryCloseDelay int = 5

var globalTime int = 0

// membranePotential += Ge / (Ge + Gi) // See page 23 file:///home/keenan/Downloads/ccnbook_ed4-3.pdf
// TODO Redo the math

type Synapse struct {
	SenderNeuron   *Neuron
	ReceiverNeuron *Neuron
	Weight         float32
}

type Neuron struct {
	Synapses            []*Synapse // Probably not needed
	MembranePotential   float32
	SpikingThisTimestep bool
	LastSpikeTimestep   int
}

type Layer struct {
	Neurons []*Neuron
}

type SimpleNetwork struct {
	Layers []*Layer
}

func UpdateLayer(layer *Layer) {
	// Lateral inhibition in a war of all against all
	numSpiking := 0
	for _, neuron := range layer.Neurons {
		if neuron.SpikingThisTimestep {
			numSpiking += 1
		}
	}
	for _, neuron := range layer.Neurons {
		if neuron.SpikingThisTimestep {
			neuron.MembranePotential -= float32(numSpiking) * inhibitoryWeight
		}
	}
}

func UpdateNeuron(neuron *Neuron) {
	if neuron.MembranePotential >= spikeThreshold {
		neuron.SpikingThisTimestep = true
		neuron.MembranePotential = 0
		neuron.LastSpikeTimestep = globalTime
	} else {
		neuron.SpikingThisTimestep = false
		neuron.MembranePotential *= leak
	}
}

// Notes for STDP:
// For each neuron, track the last time it spiked
// When a spike occurs,
// Iterate over senders, if the SN spiked recently, bump up weight by some function.
// Iterate over receivers, if the RN spiked recently, bump down weight by some function.

func UpdateSynapse(synapse *Synapse) {
	if synapse.SenderNeuron.SpikingThisTimestep {
		synapse.ReceiverNeuron.MembranePotential += synapse.Weight * weightMultiplier
	}

	// Learning
	if synapse.SenderNeuron.SpikingThisTimestep {
		timeDiff := synapse.ReceiverNeuron.LastSpikeTimestep - globalTime
		if timeDiff != 0 {
			// If timeDiff is small, that means the receiver spiked recently, and the weight should be DECREASED
			synapse.Weight -= weightChangeLearningRate / float32(timeDiff)
		}
	}
	if synapse.ReceiverNeuron.SpikingThisTimestep {
		timeDiff := synapse.SenderNeuron.LastSpikeTimestep - globalTime
		if timeDiff != 0 {
			// If timeDiff is small, that means the sender spiked recently, and the weight should be INCREASED
			synapse.Weight += weightChangeLearningRate / float32(timeDiff)
		}
	}
}

// Globally mess with variables
// increase threshold and leakiness if more neurons are spiking than desired ratio, else opposite
// TODO Maybe this should be done on a per-neuron basis too?
func RegulateHypers(network *SimpleNetwork) {
	numSpiking := 0
	numNeurons := 0 // Not optimized but IDC
	perLayerActivity := ""
	for li, layer := range network.Layers {
		numSpikingLayer := 0
		for _, neuron := range layer.Neurons {
			if neuron.SpikingThisTimestep {
				numSpiking++
				numSpikingLayer++
			}
			numNeurons++
		}
		perLayerActivity += "L" + strconv.Itoa(li) + ":" + strconv.Itoa(numSpikingLayer) + "\t"
	}
	//activeRatio := float32(numSpiking) / float32(numNeurons)
	//ratioDiff := activeRatio - targetActiveRatio // Should this be division or subtraction?
	//spikeThreshold *= 1 + (ratioDiff * regulatoryAdjustmentWeight) // Negative feedback
	println(perLayerActivity + "Active: " + strconv.Itoa(numSpiking) + " out of " + strconv.Itoa(numNeurons) + " with threshold " + strconv.FormatFloat(float64(spikeThreshold), 'f', 4, 32))
}

func main() {
	// Create patterns
	input_patterns := [][]float32{}
	output_patterns := [][]float32{}
	for i := 0; i < 25; i++ {
		pat := make([]float32, 25)
		opat := make([]float32, 25)
		for j := 0; j < 6; j++ {
			pat[(i+j)%25] = 1
			opat[(i+j+12)%25] = 1
		}
		output_patterns = append(output_patterns, opat)
		input_patterns = append(input_patterns, pat)
	}

	// Create network
	network := SimpleNetwork{}
	for l := 0; l < 4; l++ {
		lay := Layer{}
		for n := 0; n < 25; n++ {
			neur := Neuron{}
			lay.Neurons = append(lay.Neurons, &neur)
		}
		network.Layers = append(network.Layers, &lay)
	}
	// Synapses up and down
	for l := 0; l < len(network.Layers)-1; l++ {
		for _, neur1 := range network.Layers[l].Neurons {
			for _, neur2 := range network.Layers[l+1].Neurons {
				// up
				synapse := Synapse{
					SenderNeuron:   neur1,
					ReceiverNeuron: neur2,
					Weight:         rand.Float32()*0.2 + 0.9,
				}
				neur1.Synapses = append(neur1.Synapses, &synapse)
				// down
				synapse2 := Synapse{
					SenderNeuron:   neur2,
					ReceiverNeuron: neur1,
					Weight:         rand.Float32()*0.2 + 0.9,
				}
				neur2.Synapses = append(neur2.Synapses, &synapse2)
			}
		}
	}

	for epoch := 0; epoch < 2; epoch++ {
		println("EPOCH START " + strconv.Itoa(epoch))
		for patternI, inputPattern := range input_patterns {
			correctActivity := float32(0.0)
			for timestep := 0; timestep < 200; timestep++ {
				globalTime++
				for layerNum, layer := range network.Layers {
					for neuronI, neuron := range layer.Neurons {
						for _, synapse := range neuron.Synapses {
							UpdateSynapse(synapse)
						}
						UpdateNeuron(neuron)

						if layerNum == 0 && inputPattern[neuronI] == 1 {
							// Apply inputs for Input layer
							neuron.MembranePotential += weightMultiplier // TODO This isn't very principled
						}
						if layerNum == len(network.Layers)-1 {
							if output_patterns[patternI][neuronI] == 1 {
								// Apply correct outputs for Output layer in Plus Phase
								if timestep >= 150 {
									neuron.MembranePotential += weightMultiplier // This isn't very principled
								} else if timestep >= 100 {
									if neuron.SpikingThisTimestep {
										correctActivity += 1
									}
								}
							}
						}
					}
					UpdateLayer(layer)
				}

				print(strconv.Itoa(timestep) + "\t")
				RegulateHypers(&network)
			}
			println(strconv.Itoa(patternI) + "\tCorrect activity: " + strconv.FormatFloat(float64(correctActivity), 'f', 8, 32))
			// TODO Calculate and report on error
		}
	}
}
