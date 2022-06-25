import random
from collections import defaultdict

globalTime = 0

inPatternSize = 10
outPatternSize = 10
numPatterns = 10

durationOneTimestep = 1 # 1ms

# Activity Hyperparams
startingSpikeThreshold = 0.1 # Note that this specific to each neuron is changed by the thresholdNudgeVelocity to bring Neurons more inline with the desiredInterSpikeInterval
desiredInterSpikeInterval = 30
thresholdNudgeVelocity = 0.0 #000001 # This should be slow, so that not much change occurs within a trial
thresholdDecay = 1.0 #0.99 # I'm not sure why this is needed, but otherwise Thresholds seem to blow up
# topKHidden = 10 # If this value is greater than 0, only the top k hidden neurons will fire, as measured by membrane potential over threshold

# Timescale Hyperparams
refractoryPeriod = 3 # Can't spike more often than this
activityDecayPerMillisecond = 0.85
activityLinearDecayPerMillisecond = 0.01

# Learning Hyperparams
learningRate = 0.0001
enforcedWeightAverage = 0.01 # Should be much bigger than learningRate

runningLists = defaultdict(list)
def TallyItemForRunningAverage(name, item, length=100):
    # This could be optimized by using some other data structure, but ¯\_(ツ)_/¯
    runningLists[name].append(item)
    if len(runningLists[name]) > length:
        runningLists[name] = runningLists[name][len(runningLists[name]) - length:]

def AverageFromTally(name, eps=0.01):
    return (sum(runningLists[name]) + eps) / (len(runningLists[name]) + eps)

def VarianceFromTally(name, eps=1):
    mean = (sum(runningLists[name]) + eps) / (len(runningLists[name]) + eps)
    return (sum((x - mean) ** 2 for x in runningLists[name]) + eps) / (len(runningLists[name]) + eps)

def StdDevFromTally(name, eps=1):
    mean = (sum(runningLists[name]) + eps) / (len(runningLists[name]) + eps)
    return ((sum((x - mean) ** 2 for x in runningLists[name]) + eps) / (len(runningLists[name]) + eps)) ** 0.5

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
        self.lastWeightAverage = enforcedWeightAverage # This holds a number from one update to the next

        # Fully connected
        self.weights = [[random.random() * .2]*self.size for _ in range(self.size)]

        self.timeWithinTrial = 0.0

        # Just for dev and debugging
        self.curInputs = []
        self.curOutputs = []
        self.totalAdjustments = [[0.0] * self.size for _ in range(self.size)]

    def StartNewTrial(self):
        self.membranePotentials = [random.random() * 0.1 for _ in range(self.size)]
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
                self.membranePotentials[start + i] = self.spikingThreshold[start + i] + 1.0  # TODO This might not be the most elegant way to clamp it
            else:
                self.membranePotentials[start + i] = self.spikingThreshold[start + i] - 1.0

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

    def DecayMembranePotentials(self):
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
            for i in range(self.size):
                if self.currentlySpiking[i]:
                    self.minusPhaseSpikeCounts[i] += 1
        else: # Plus phase
            for i in range(self.size):
                if self.currentlySpiking[i]:
                    self.plusPhaseSpikeCounts[i] += 1

    def UpdateWeights(self):
        minusSum = 0.0
        plusSum = 0.0
        weightSum = 0.0
        totalActivitySum = 0.0
        count = 0
        minusAve = AverageFromTally("minus", 1)
        plusAve = AverageFromTally("plus", 1)
        minusVar = VarianceFromTally("minus", 1)
        plusVar = VarianceFromTally("plus", 1)
        adjustments = [[0]*self.size for _ in range(self.size)]
        for sendI in range(self.size):
            for recI in range(self.size):
                if sendI == recI: # No self connections. Also note that this learning rule is symmetrical, so we could optimize with <
                    self.weights[sendI][recI] = 0 # No self weights
                else:
                    # TODO Multiplication not subtraction
                    sendMinusPhaseActivity = float(self.minusPhaseSpikeCounts[sendI] / 3) # Divide by 3 because the minus phase is 150ms and the plus phase is 50ms
                    sendPlusPhaseActivity = float(self.plusPhaseSpikeCounts[sendI])
                    recMinusPhaseActivity = float(self.minusPhaseSpikeCounts[recI] / 3)
                    recPlusPhaseActivity = float(self.plusPhaseSpikeCounts[recI])

                    # This is the core learning rule. Note that it's symmetric
                    # adjustment = (sendMinusPhaseActivity - minusAve) * (recPlusPhaseActivity - plusAve) + (recMinusPhaseActivity - minusAve) * (sendPlusPhaseActivity - plusAve) # I think this is wrong
                    # adjustment = (recPlusPhaseActivity - plusAve) * (sendPlusPhaseActivity - plusAve) # Just look at plus phase
                    adjustment = ((sendPlusPhaseActivity - plusAve) * (recPlusPhaseActivity - plusAve)) + ((sendMinusPhaseActivity - minusAve) * (recMinusPhaseActivity - minusAve)) # https://raw.githubusercontent.com/CompCogNeuro/ed4/master/ccnbook_ed4.pdf Page 73
                    adjustments[sendI][recI] = "{:.3f}".format(adjustment * learningRate)
                    self.totalAdjustments[sendI][recI] += adjustment * learningRate

                    if recPlusPhaseActivity > 0 or sendPlusPhaseActivity > 0:
                        x = 5
                        #print("Activity at: ", recI, sendI, adjustment)# recPlusPhaseActivity, sendPlusPhaseActivity, plusAve)
                    if recPlusPhaseActivity > 0 and sendPlusPhaseActivity > 0:
                        x = 5

                    # For normalization
                    minusSum += sendMinusPhaseActivity + recMinusPhaseActivity
                    plusSum += sendPlusPhaseActivity + recPlusPhaseActivity
                    count += 1
                    weightSum += self.weights[sendI][recI]

                    if self.disableLearning: # During warm up we don't adjust weights, but do warm up other statistics
                        continue
                    # Enforce weights to be non-negative
                    self.weights[sendI][recI] = max(0.0, self.weights[sendI][recI] + adjustment * learningRate)
                    self.weights[sendI][recI] *= (enforcedWeightAverage / self.lastWeightAverage) # Enforce the weights to be the average we want

        TallyItemForRunningAverage("minus", minusSum / count, 100)
        TallyItemForRunningAverage("plus", plusSum / count, 100)
        TallyItemForRunningAverage("weight", weightSum / count, 100)
        self.lastWeightAverage = weightSum / count

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
        thresholdSum = 0.0
        numSpiking = 0.0
        count = 0
        for i in range(self.size):
            thresholdSum += self.spikingThreshold[i]
            if self.currentlySpiking[i]:
                # Asymmetric to target this ratio
                self.spikingThreshold[i] += thresholdNudgeVelocity * desiredInterSpikeInterval
                numSpiking += 1
            else:
                self.spikingThreshold[i] -= thresholdNudgeVelocity
            self.spikingThreshold[i] *= thresholdDecay
            count += 1
        TallyItemForRunningAverage("threshold", thresholdSum / count)
        TallyItemForRunningAverage("spiking perc", numSpiking / count)

    def Diagnostics(self):
        for i in range(self.size):
            self.recentActivity[i] = 1.0 if self.timeLastSpiked[i] + refractoryPeriod >= self.timeWithinTrial else 0.0

    def PrintState(self):
        # s = ''.join("*" if spiking else "_" for spiking in self.currentlySpiking) # Spikes
        s = ''.join("A" if act > 0.5 else "_" for act in self.recentActivity) # Activity
        s = "I:" + s[0: self.inputSize] + "O:" + s[self.inputSize: self.inputSize + self.outputSize] + "H:" + s[self.inputSize + self.outputSize:]
        stats = "\tAveThresh: " + "{:.2f}".format(AverageFromTally("threshold")) + "\tSpikingPerc: " + "{:.2f}".format(AverageFromTally("spiking perc")) + "\tAveWeight: " + "{:.2f}".format(AverageFromTally("weight"))
        return s + stats

    def UpdateOneTimestep(self):
        self.DetermineSpikers() # Because this happens at the start of this function, it occurs right after inputs and outputs are applied
        self.DecayMembranePotentials()
        self.PropagateSpikes()
        self.RecordForLearning()
        self.ApplyRegulation()
        self.Diagnostics()

    def OneTrial(self, inputs, outputs, printEveryStep=False):
        self.curInputs = inputs
        self.curOutputs = outputs
        self.StartNewTrial()
        for i in range(0, 200, durationOneTimestep):
            self.timeWithinTrial = i
            self.ApplyInputOrOutput(inputs, 0)
            if self.timeWithinTrial >= 150:
                self.ApplyInputOrOutput(outputs, self.inputSize)
            self.UpdateOneTimestep()
            self.numCorrectThisTrial += self.EvaluatePerformance(outputs)
            if printEveryStep:
                print("Millisecond: ", i, "\t", self.PrintState())
        self.UpdateWeights()

network = Network(inPatternSize, outPatternSize, 0)

patternDensity = 0.2
# patterns = [([1 if random.random() > 1-patternDensity else 0 for _ in range(inPatternSize)], [-1 for _ in range(outPatternSize)]) for _ in range(numPatterns)]
patterns = [([1 if i==j else 0 for i in range(inPatternSize)], [0 for _ in range(outPatternSize)]) for j in range(numPatterns)] # Only one neuron is active at a time
patterns = [(patterns[i][0], patterns[i][0]) for i in range(len(patterns))] # Make inputs and outputs the same to make it easier to learn

warmupPatterns = [([1 if random.random() > 1-patternDensity else 0 for _ in range(inPatternSize)], [1 if random.random() > 1-patternDensity else 0 for _ in range(outPatternSize)]) for _ in range(numPatterns)]

print(patterns)

network.disableLearning = True
for warmupEpoch in range(10):
    patNum = 0
    for (input, output) in warmupPatterns:
        network.OneTrial(input, output, )
        print("Warmup: ", warmupEpoch, "\tPattern: ", patNum, "\t", network.PrintState(), "\tCorrect: ", int(network.numCorrectThisTrial))
        patNum += 1

network.disableLearning = False
evaluationsPerEpoch = []
for epoch in range(50):
    patNum = 0
    correctOverallEpoch = 0.0
    for (input, output) in patterns:
        network.OneTrial(input, output)
        print("Epoch:  ", epoch, "\tPattern: ", patNum, "\t", network.PrintState(), "\tCorrect: ", int(network.numCorrectThisTrial))
        correctOverallEpoch += network.numCorrectThisTrial
        patNum += 1
    evaluationsPerEpoch.append(int(correctOverallEpoch))

print("\nPrinting example trial with Input: ", patterns[0][0], "Outputs: ", patterns[0][1])
network.OneTrial(patterns[0][0], patterns[0][1], printEveryStep=True)

print("\nEvaluations: ", evaluationsPerEpoch)

# Print weights
print("\nWeights:")
weightDisplay = ""
for i, ws in enumerate(network.weights):
    for j, w in enumerate(ws):
        if i == j:
            weightDisplay += "_____ "
        elif i+inPatternSize == j or i-inPatternSize == j:
            weightDisplay += "{:.3f}".format(w) + "<"
        else:
            weightDisplay += "{:.3f}".format(w) + " "
    weightDisplay += "\n"
print(weightDisplay)

print("Thresholds: ", network.spikingThreshold)

print("end")












