import matplotlib.pyplot as plt
import math

interval_number = []
counts = []
file_name = 'Background Radiation.txt'
with open(file_name) as file:
    # skip first two lines
    file.readline()
    file.readline()
    for line in file:
        interval_number.append(float(line.split()[0]))
        counts.append(float(line.split()[1]))
counts = [n for n in counts]
interval_number = [n / 3 for n in interval_number]

const = sum(counts) / len(counts)

# print(math.sqrt(sum(counts)) / len(counts))

'''
print(const)

yerr = [math.sqrt(n) for n in counts]

plt.errorbar(interval_number, counts, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('Counts of Background Radiation over Time', size = 20)
plt.ylabel('Number of Counts per 20 Second Interval', size = 15)
plt.xlabel('Time (minutes)', size = 15)
plt.show()

delta_n2 = [1 / (n ** 2) for n in yerr]
sigma = math.sqrt(1 / sum(delta_n2))
print(sigma)

counts = [n - const for n in counts]
plt.errorbar(interval_number, counts, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('Residual Counts of Background Radiation over Time', size = 20)
plt.ylabel('Number of Residual Counts per 20 Second Interval', size = 15)
plt.xlabel('Time (minutes)', size = 15)
plt.show()
'''