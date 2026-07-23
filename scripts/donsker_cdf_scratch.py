# import numpy as np
# import matplotlib.pyplot as plt

# X1 = np.random.uniform(0,1, 50)
# X2 = np.random.uniform(0,1, 500)
# X3 = np.random.uniform(0,1, 5000)

# def emp_dist(X, x):
#     N = len(X)
#     return (1/np.sqrt(N)) * (np.sum(np.where(X<= x, 1, 0)) - N * x) 

# X = np.arange(0,1,.0001)
# Y1 = [emp_dist(X1, i) for i in X]
# Y2 = [emp_dist(X2, i) for i in X]
# Y3 = [emp_dist(X3, i) for i in X]



# plt.title('test')
# plt.subplot(3, 1, 1)
# plt.plot(X, Y1, lw = .5, c = 'C0')
# plt.subplot(3, 1, 2)
# plt.plot(X, Y2, lw = .5, c = 'C0')
# plt.subplot(3, 1, 3)
# plt.plot(X, Y3, lw = .5, c = 'C0')

import numpy as np
import matplotlib.pyplot as plt

# Sample sizes

X3 = np.random.uniform(0, 1, 1000)
X1 = X3[::20]
X2 = X3[::8]

def emp_dist(X, x):
    N = len(X)
    return (1 / np.sqrt(N)) * (np.sum(X <= x) - N * x)

X = np.linspace(0, 1, 1000)
Y1 = [emp_dist(X1, xi) for xi in X]
Y2 = [emp_dist(X2, xi) for xi in X]
Y3 = [emp_dist(X3, xi) for xi in X]

fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

axs[0].plot(X, Y1, lw=0.8, c='C0')
axs[0].axhline(0, color='black', lw=0.5, linestyle='--')
# axs[0].set_title(r'$N = 40$')

axs[1].plot(X, Y2, lw=0.8, c='C0')
axs[1].axhline(0, color='black', lw=0.5, linestyle='--')
# axs[1].set_title(r'$N = 200$')

axs[2].plot(X, Y3, lw=0.8, c='C0')
axs[2].axhline(0, color='black', lw=0.5, linestyle='--')
# axs[2].set_title(r'$N = 1000$')

# for i in range(1000):
#     Y0 = np.random.uniform(0, 1, 1000)
#     Y0 = [emp_dist(Y0, xi) for xi in X]
#     axs[3].plot(X, Y0, lw=0.8, c='C0', alpha = .1)
#     axs[3].axhline(0, color='black', lw=0.5, linestyle='--')

# axs[1].set_ylabel(r'$\sqrt{N}(F_N(x) - F(x))$')

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-1.2,1.2])
    # ax.set_ylabel(r'$\sqrt{N}(F_N(x) - F(x))$')
    # ax.grid(True)

# axs[2].set_xlabel(r'$x$')

plt.subplots_adjust(hspace=0)
# plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

