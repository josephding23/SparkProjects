import matplotlib.pyplot as plt
import numpy as np
plt.ioff()
for i in range(15):
    plt.plot(np.random.rand(10))
plt.show()