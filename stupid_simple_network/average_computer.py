from collections import defaultdict

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
