
import random

emotions = ["tired", "angry", "friendly"]
behaviors = ["sleeping", "eating", "playful", "hiding"]


emo = [[.6, .2, .2], [.4, .5, .1], [.1, .1, .8]]
behavo = [[.7, .1, 0, .2], [.2, .3, .1, .4], [.2, .1, .7, 0]]

e = 0
b = 0

for i in range(10000):
    # get next emotion
    r = random.random()
    c = 0
    for j in range(len(emo[e])):
        c += emo[e][j]
        if r < c:
            e = j
            break
    # print(emotions[e])
    c = 0
    for j in range(len(behavo[e])):
        c += behavo[e][j]
        if r < c:
            b = j
            break
    if random.random() < .1:
        # narcolepsy
        b = 0
        e = 0
    if random.random() < .1:
        # playful
        b = 2
        e = 2
    print(behaviors[b])