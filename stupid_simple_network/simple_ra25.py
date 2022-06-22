import random

# Notes: Kevin did something like this and got it to learn simple I/O pairs

# Neurons:
# Excitatory conductance, inhib conduct, leak,
# those feed into membrane potential,
# spiking when membrane potential is over a threshold

# Try constant lateral inhibition

# Learning can be hebbian or STDP or whatever
# can do the kinase learning thing whatever

# Later: lateral inhib, NMDA, frequency regulation

# Hyperparameters
# leak = 0.8
# weightChangeLearningRate = 0.01
# deltaTime = 1 # in milliseconds
# inhibitoryWeight = 0.1
# targetActiveRatio = 0.20          # This probably has a big effect
# regulatoryAdjustmentWeight = 0.01 # Should be like 0.01
# excitatoryCloseDelay = 5
# inhibitoryCloseDelay = 5

globalTime = 0

inPatternSize = 25
outPatternSize = 25
numPatterns = 10

durationOneTimestep = 1 # 1ms

# Activity Hyperparams
startingSpikeThreshold = 1.0 # Note that this specific to each neuron is changed by the thresholdNudgeVelocity to bring Neurons more inline with the desiredInterSpikeInterval
desiredInterSpikeInterval = 10
thresholdNudgeVelocity = 0.01 # This should be slow, so that not much change occurs within a trial
# topKHidden = 10 # If this value is greater than 0, only the top k hidden neurons will fire, as measured by membrane potential over threshold
activityDecayPerMillisecond = .9

# Timescale Hyperparams
refractoryPeriod = 3 # Can't spike more often than this

# Learning Hyperparams
learningRate = 0.01

class Network:
    def __init__(self, inputSize: int, outputSize: int, hiddenSize: int):
        self.size = inputSize + outputSize + hiddenSize
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.StartNewTrial()
        # All this stuff is done with matrices rather than objects so that in the future it can be more easily parallelized
        self.spikingThreshold = [startingSpikeThreshold for _ in range(self.size)]
        self.disableLearning = False

        # Fully connected
        self.weights = [[random.random() * .2]*self.size for _ in range(self.size)]

        self.timeWithinTrial = 0.0

    def StartNewTrial(self):
        self.membranePotentials = [random.random() for _ in range(self.size)]
        self.timeLastSpiked = [random.randint(-4, -1) for _ in range(self.size)] # Some randomization
        self.currentlySpiking = [False for _ in range(self.size)]
        self.recentActivity = [0.0 for _ in range(self.size)] # This is purely for viewing and diagnostics

        # For learning
        self.minusPhaseSpikeCounts = [0] * self.size # Count spikes by sending neuron
        self.plusPhaseSpikeCounts = [0] * self.size # Count spikes by receiving neuron
        self.numCorrectThisTrial = 0

    def ApplyInputOrOutput(self, patterns, start):
        for i in range(0, len(patterns)): # Inputs start at 0
            if patterns[i] > 0.5:
                self.membranePotentials[start + i] = self.spikingThreshold[i] + 1.0  # TODO This might not be the most elegant way to clamp it

    def DetermineSpikers(self):
        for i in range(self.size):
            if self.membranePotentials[i] >= self.spikingThreshold[i] and self.timeLastSpiked[i] + refractoryPeriod <= self.timeWithinTrial:
                self.currentlySpiking[i] = True
            else:
                self.currentlySpiking[i] = False

        # # Top K
        # if False and topKHidden > 0:
        #     for i in range(self.inputSize + self.outputSize, self.size):
        #         self.currentlySpiking[i] = False
        #     hiddenSpikers = sorted([(self.membranePotentials[i] - self.spikingThreshold[i], i) for i in range(self.inputSize + self.outputSize, self.size)], reverse=True, key=lambda x: x[0])[:topKHidden] # Wow this is slow lol
        #     for (value, i) in hiddenSpikers:
        #         self.currentlySpiking[i] = True

        # Some bookkeeping for spikers
        for i in range(self.size):
            if self.currentlySpiking[i]:
                self.timeLastSpiked[i] = self.timeWithinTrial
                self.membranePotentials[i] = 0.0

    def DecayWeights(self):
        for i in range(self.size):
            self.membranePotentials[i] *= activityDecayPerMillisecond

    def PropagateSpikes(self):
        for sendI in range(self.size):
            if self.currentlySpiking[sendI]:
                for recI in range(self.size):
                    if sendI != recI: # No self connections
                        self.membranePotentials[recI] += self.weights[sendI][recI]

    def RecordForLearning(self):
        if self.timeWithinTrial < 150: # Minus phase
            for sendI in range(self.size):
                if self.currentlySpiking[sendI]:
                    self.minusPhaseSpikeCounts[sendI] += 1
        else: # Plus phase
            for recI in range(self.size):
                if self.currentlySpiking[recI]:
                    self.plusPhaseSpikeCounts[recI] += 1

    def UpdateWeights(self):
        if self.disableLearning:
            return
        for sendI in range(self.size):
            for recI in range(self.size):
                if sendI != recI: # No self connections
                    adjustment = ((self.minusPhaseSpikeCounts[sendI] / 3) - self.plusPhaseSpikeCounts[recI]) * learningRate
                    self.weights[sendI][recI] = max(0.0, self.weights[sendI][recI] + adjustment)

    def EvaluatePerformance(self, outputs):
        numCorrect = 0.0
        if self.timeWithinTrial >= 100 and self.timeWithinTrial < 150: # Only evaluate during the latter part of the minus phase
            for i in range(0, self.outputSize):
                # activeI = self.currentlySpiking[self.inputSize + i]
                activeI = bool(self.recentActivity[self.inputSize + i])
                if outputs[i] == activeI:
                    numCorrect += 1
        return numCorrect / 50 # 50 = 150 - 100

    def ApplyRegulation(self):
        # Adjust threshold on a neuron-by-neuron basis
        for i in range(self.size):
            if self.currentlySpiking[i]:
                # Asymmetric to target this ratio
                self.spikingThreshold[i] += thresholdNudgeVelocity * desiredInterSpikeInterval
            else:
                self.spikingThreshold[i] -= thresholdNudgeVelocity

    def Diagnostics(self):
        for i in range(self.size):
            self.recentActivity[i] = 1.0 if self.timeLastSpiked[i] + refractoryPeriod >= self.timeWithinTrial else 0.0

    def PrintState(self):
        s = ''.join("*" if spiking else "_" for spiking in self.currentlySpiking) # Spikes
        # s = ''.join("A" if act > 0.5 else "_" for act in self.recentActivity) # Activity
        s = "I:" + s[0: self.inputSize] + "O:" + s[self.inputSize: self.inputSize + self.outputSize] + "H:" + s[self.inputSize + self.outputSize:]
        return s

    def UpdateOneTimestep(self):
        self.DetermineSpikers()
        self.PropagateSpikes()
        self.RecordForLearning()
        self.ApplyRegulation()
        self.Diagnostics()

    def OneTrial(self, inputs, outputs, printEveryStep=False):
        self.StartNewTrial()
        for i in range(0, 200, durationOneTimestep):
            self.timeWithinTrial = i
            self.ApplyInputOrOutput(inputs, 0)
            if self.timeWithinTrial >= 150:
                self.ApplyInputOrOutput(outputs, self.inputSize)
            self.UpdateOneTimestep()
            self.numCorrectThisTrial += self.EvaluatePerformance(outputs)
            if printEveryStep:
                print("Timestep: ", i, "\t", self.PrintState())
        self.UpdateWeights()

network = Network(inPatternSize, outPatternSize, 0)

patternDensity = 0.2
patterns = [([1 if random.random() > 1-patternDensity else 0 for _ in range(inPatternSize)], [-1 for _ in range(outPatternSize)]) for _ in range(numPatterns)]
patterns = [(patterns[i][0], patterns[i][0]) for i in range(len(patterns))] # Make inputs and outputs the same to make it easier to learn

warmupPatterns = [([1 if random.random() > 1-patternDensity else 0 for _ in range(inPatternSize)], [1 if random.random() > 1-patternDensity else 0 for _ in range(outPatternSize)]) for _ in range(numPatterns)]

print(patterns)

network.disableLearning = True
for warmupEpoch in range(2):
    patNum = 0
    for (input, output) in warmupPatterns:
        network.OneTrial(input, output, )
        print("Warmup: ", warmupEpoch, "\tPattern: ", patNum, "\t", network.PrintState(), "\tCorrect: ", int(network.numCorrectThisTrial))
        patNum += 1

network.disableLearning = False
evaluationsPerEpoch = []
for epoch in range(100):
    patNum = 0
    correctOverallEpoch = 0.0
    for (input, output) in patterns:
        network.OneTrial(input, output)
        print("Epoch:  ", epoch, "\tPattern: ", patNum, "\t", network.PrintState(), "\tCorrect: ", int(network.numCorrectThisTrial))
        correctOverallEpoch += network.numCorrectThisTrial
        patNum += 1
    evaluationsPerEpoch.append(int(correctOverallEpoch))

print("\nPrinting example trial")
network.OneTrial(patterns[0][0], patterns[0][1], printEveryStep=True)

print("\nEvaluations: ", evaluationsPerEpoch)
















