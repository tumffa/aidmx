import statistics
import math

# Given data
data = [104, 110, 114, 97, 105, 113, 106, 101, 100, 107]

# Calculate mean and standard deviation
mean = 106
std_dev = statistics.stdev(data)
sum = 0
for i in data:
    sum += (i - mean) ** 2
print("Sum:", sum)
sd = math.sqrt(sum / (len(data) - 1))
print("Standard Deviation:", sd)
print("Mean:", mean)
