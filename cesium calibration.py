import math
import matplotlib.pyplot as plt

y = [0, 1042, 1961, 2141, 2059, 2235, 2156, 2132, 2227, 2162, 2232, 2240, 2300, 2250]
x = [550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]

y_err = [math.sqrt(n) for n in y]

plt.errorbar(x, y, yerr = y_err, xerr = 5, marker='.', linestyle = 'none', ecolor = 'orange', capsize = 1.5)

plt.title('Voltage versus Counts', size = 20)
plt.xlabel('Voltage (Volts)', size = 15)
plt.ylabel('Number of Counts per Standard Time', size = 15)
plt.show()

