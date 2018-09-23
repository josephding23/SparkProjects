import matplotlib.pyplot as plt
from matplotlib.pylab import hist
import numpy as np

def plot_gallery(images, h, w, n_row=2, n_col=5):
	plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
	plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
	for i in range(n_row * n_col):
		plt.subplot(n_row, n_col, i+1)
		plt.imshow(images[:, i].reshape((h, w)), cmap=plt.cm.gray)
		plt.title("Eigenface %d" % (i+1), size=12)
		plt.xticks(())
		plt.yticks(())

pcs = np.loadtxt("./tmp/pc.csv", delimiter=",")
plot_gallery(pcs, 50, 50)