import math
from scipy.optimize import curve_fit
from background import const
import matplotlib.pyplot as plt
import numpy as np

# do double exponential fit A + B * e  + C * e


# b represent Pb-214 and c represent Bi-214
ec = 0.95
eb = 0.80
lb = math.log(2) / (27.0 * 60)
lc = math.log(2) / (19.9 * 60)

interval_number = []
counts = []
file_name = 'air_sample.txt'
with open(file_name) as file:
    # skip first two lines
    file.readline()
    file.readline()
    for line in file:
        interval_number.append(float(line.split()[0]))
        counts.append(float(line.split()[1]))

time = [20 * x  for x in interval_number]
counts = [x - const for x in counts]

s_time = time[90:]
# s_time = [math.log(x, 2) for x in s_time]
# s_time = [2 ** (x / 2000) for x in s_time]
s_counts = counts[90:]
yerr = [math.sqrt(n) for n in s_counts]
s_counts = [math.log(x, 2) for x in s_counts]
yerr = [1 / math.log(x, 2) for x in yerr]
def linear(t, m, b):
    return b - m * t;

popt, pconv = curve_fit(linear, s_time, s_counts)
m, b = popt

print(m)
print(b)
print(pconv)

counts_pred = []
for i in range(len(s_time)):
    counts_pred.append(b - m * s_time[i])

chi_square = 0
for i in range(len(s_time)):
    chi_square = chi_square + ((s_counts[i] - counts_pred[i]) ** 2) / (yerr[i] ** 2)
reduced_chi_square = chi_square / (len(s_time) - 2)
print(reduced_chi_square)

plt.errorbar(s_time, s_counts, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.plot(s_time, counts_pred)
plt.title('Semilog Plot of Counts over Time with Linear FIt', size = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Seconds)', size = 15)
plt.ylabel('Number of Counts per 20 Seconds Interval on a Logarithmic Scale of Base 2', size = 15)
plt.show()

s_counts_residual = []
for i in range(len(s_time)):
    s_counts_residual.append(s_counts[i] - counts_pred[i])

plt.errorbar(s_time, s_counts_residual, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.title('Residuals of Counts over Time with Semilog Plot', size = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Seconds)', size = 15)
plt.ylabel('Residual of Number of Counts per 20 Seconds Interval on a Logarithmic Scale of Base 2', size = 15)
plt.show()

'''
print(1 / m)

counts_pred = []
for i in range(len(time)):
    counts_pred.append(2 ** (b - m * time[i]))

yerr = [math.sqrt(n + const) for n in counts]

plt.errorbar(time, counts, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.plot(time, counts_pred)
plt.title('Extrapolating Semilog Linear Fit to Whole Graph', size = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Seconds)', size = 15)
plt.ylabel('Number of Counts per 20 Seconds Interval', size = 15)
plt.show()
filtered_counts = []
for i in range(len(time)):
    filtered_counts.append(counts[i] - counts_pred[i])

plt.errorbar(time[0:90], filtered_counts[0:90], yerr = yerr[0:90], marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.title('Residuals of Semilog Fitting on Whole Graph', size = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Seconds)', size = 15)
plt.ylabel('Residual of Counts per20 Second Interval', size = 15)
plt.show()
'''
'''
s_counts_pred = []
for i in range(len(s_time)):
    s_counts_pred.append(linear(s_time[i], m, b))

filter = []
for i in range(len(time)):
    filter.append(2 ** (b - m * time[i]))
    # filter.append(b + m * 2 ** (time[i] / 2000))
    # filter.append(2 ** (b + m * time[i]))

for i in range(len(time)):
    counts[i] = counts[i] - filter[i]
'''

# print(counts)
def objective(t, R, B_obs_0):
    return B_obs_0 * ((1 + (ec / eb) * lc / (lc - lb)) * math.e ** (- lb * np.array(t)) + (ec / eb) * (R -  lc / (lc - lb)) * math.e ** (- lc * np.array(t)))
popt, pcov = curve_fit(objective, time, counts, [1.0, 118])
R, B_obs_0 = popt
print(pcov)

counts_pred = objective(time, R, B_obs_0)

yerr = [math.sqrt(n + const) for n in counts]

plt.errorbar(time, counts, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.plot(time, counts_pred)
plt.title('Counts per 20 Second Interval versus Time Fitted to Model', size = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Seconds)', size = 15)
plt.ylabel('Counts per 20 Second Interval', size = 15)
plt.show()

print(R)
print(B_obs_0)

chi_square = 0
for i in range(len(time)):
    chi_square = chi_square + ((counts[i] - counts_pred[i]) ** 2) / (yerr[i] ** 2)
reduced_chi_square = chi_square / (len(time) - 2)
print(reduced_chi_square)

diff = []
for i in range(len(time)):
    diff.append(counts[i] - counts_pred[i])

plt.errorbar(time, diff, yerr = yerr, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)
plt.title('Residual of Counts per 20 Second Interval versus Time', size = 20)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel('Time (Seconds)', size = 15)
plt.ylabel('Residual of Counts per 20 Second Interval', size = 15)
plt.show()

'''
def objective_2(t, A, B, C):
    return A + B * math.e ** (- np.array(t) * lb) + C * math.e ** (- np.array(t) * lc)

popt, _ = curve_fit(objective_2, time, counts)
A, B, C = popt

counts_pred = objective_2(time, A, B, C)

plt.scatter(time, counts)
plt.plot(time, counts_pred)
plt.show()

R = (C / B) * (1 + (ec / eb) * lc / (lc - lb)) + lc / (lc - lb)
B_obs_0 = B / (1 + (ec / eb) * lc / (lc - lb))
print(R)
print(B_obs_0)

diff = []
for i in range(len(time)):
    diff.append(counts[i] - counts_pred[i])

plt.scatter(time, diff)
plt.show()

for i in range(len(time)):
    diff[i] = diff[i] / math.sqrt(counts[i] + const)

plt.scatter(time, diff)
plt.show()
'''

# you should at least mention the more complex model

