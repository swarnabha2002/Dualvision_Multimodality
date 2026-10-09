import os
os.chdir(r"E:\dualvision\DualVision")

test = sorted(int(l.strip().split("/")[-1].split(".")[0])
              for l in open("data/M3FD/splits/test.txt") if l.strip())

runs = []
start = prev = test[0]
for n in test[1:]:
    if n == prev + 1:
        prev = n
    else:
        runs.append((start, prev))
        start = prev = n
runs.append((start, prev))

lens = [b - a + 1 for a, b in runs]
print("test images :", len(test))
print("runs        :", len(runs))
print("median run  :", sorted(lens)[len(lens)//2])
print("singletons  :", sum(1 for x in lens if x == 1))
print("first 15    :", runs[:15])