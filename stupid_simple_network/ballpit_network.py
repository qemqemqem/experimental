import random
from collections import defaultdict
from average_computer import TallyItemForRunningAverage, AverageFromTally, VarianceFromTally, StdDevFromTally

# The conceit of this file is that I'm pretending that spikes are like balls that move around. They're conserved

globalTime = 0

# Pattern parameters
inPatternSize = 10
outPatternSize = 10
numPatterns = 10

durationOneTimestep = 1 # 1ms

totalNumBalls = 100
pathsPerNode = 9
ballsNeededToSpike = 4

# TODO Implement
ballDecayRate = 20000000 # Lose one ball every 4 time units

class Network:
	def __init__(self, inputSize: int, outputSize: int, hiddenSize: int):
		self.size = inputSize + outputSize + hiddenSize
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.hiddenSize = hiddenSize
		self.disableLearning = False
		self.timeWithinTrial = 0.0

		# For connections and stuff
		self.pathways = [[] for _ in range(self.size)] # These are like weights
		self.ballsInReserve = totalNumBalls
		self.numBalls = [0] * self.size
		self.pathsInReserve = [pathsPerNode] * self.size
		for i in range(self.size):
			for j in range(self.pathsInReserve[i]):
				self.pathways[i].append(random.randint(0, self.size - 1)) # Randomly distribute paths
			self.pathsInReserve[i] = 0

		# Just for dev and debugging
		self.curInputs = []
		self.curOutputs = []
		self.totalAdjustments = [[0.0] * self.size for _ in range(self.size)]

		self.StartNewTrial()

	def StartNewTrial(self):
		self.ballsNeededToSpike = [ballsNeededToSpike for _ in range(self.size)]
		self.currentlySpiking = [False for _ in range(self.size)]

		while self.ballsInReserve > 0:
			self.TryAddBall(random.randint(0, self.size - 1))

		# # For learning
		# self.minusPhaseSpikeCounts = [0] * self.size # Count spikes by sending neuron
		# self.plusPhaseSpikeCounts = [0] * self.size # Count spikes by receiving neuron
		self.numCorrectThisTrial = 0

	def TryAddBall(self, i):
		if self.ballsInReserve > 0:
			self.ballsInReserve -= 1
			self.numBalls[i] += 1

	def TryRemoveBall(self, i):
		if self.numBalls[i] > 0:
			self.ballsInReserve += 1
			self.numBalls[i] -= 1

	def TryMoveBall(self, fm, to):
		if self.numBalls[fm] > 0:
			self.numBalls[fm] -= 1
			self.numBalls[to] += 1

	def ApplyInputOrOutput(self, patterns, start):
		for i in range(len(patterns)):
			if patterns[i] > 0.5:
				self.TryAddBall(start + i)
			else:
				self.TryRemoveBall(start + i)

	def DetermineSpikers(self):
		for i in range(self.size):
			if self.numBalls[i] >= self.ballsNeededToSpike[i]:
				self.currentlySpiking[i] = True
				# self.ballsInReserve += self.numBalls[i] # These get dispersed during the propagation
				# self.numBalls[i] = 0
			else:
				self.currentlySpiking[i] = False

	def DecayMembranePotentials(self):
		for i in range(self.size):
			if random.random() < 1.0 / float(ballDecayRate): # TODO Maybe this shouldn't be random??
				self.TryRemoveBall(i)

	def PropagateSpikes(self):
		for sendI in range(self.size):
			if self.currentlySpiking[sendI]:
				while self.numBalls[sendI] > 0:
					# TODO I'm concerned that this randomness might be bad but idk it's probably fine with a large number of balls
					self.TryMoveBall(sendI, random.choice(self.pathways[sendI]))

	def RecordForLearning(self):
		pass

	def UpdateWeights(self):
		pass

	def EvaluatePerformance(self, outputs):
		numCorrect = 0.0
		if self.timeWithinTrial >= 100 and self.timeWithinTrial < 150: # Only evaluate during the latter part of the minus phase
			for i in range(0, self.outputSize):
				activeI = self.currentlySpiking[self.inputSize + i]
				if outputs[i] == activeI:
					numCorrect += 1
		return numCorrect / 50 # 50 = 150 - 100

	def ApplyRegulation(self):
		pass

	def Diagnostics(self):
		pass

	def PrintState(self):
		s = ''.join("*" if spiking else "_" for spiking in self.currentlySpiking) # Spikes
		# s = ''.join("A" if act > 0.5 else "_" for act in self.recentActivity) # Activity
		s = "I:" + s[0: self.inputSize] + "O:" + s[self.inputSize: self.inputSize + self.outputSize] + "H:" + s[self.inputSize + self.outputSize:]
		# stats = "\tAveThresh: " + "{:.2f}".format(AverageFromTally("threshold")) + "\tSpikingPerc: " + "{:.2f}".format(AverageFromTally("spiking perc")) + "\tAveWeight: " + "{:.2f}".format(AverageFromTally("weight"))
		stats = "\tReserveBalls: " + str(self.ballsInReserve)
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
for warmupEpoch in range(1):
	patNum = 0
	for (input, output) in warmupPatterns:
		network.OneTrial(input, output, )
		print("Warmup: ", warmupEpoch, "\tPattern: ", patNum, "\t", network.PrintState(), "\tCorrect: ", int(network.numCorrectThisTrial))
		patNum += 1

network.disableLearning = False
evaluationsPerEpoch = []
for epoch in range(2):
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

















