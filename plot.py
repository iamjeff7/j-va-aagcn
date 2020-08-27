import matplotlib.pyplot as plt
import numpy as np

def line_chart(filename):
    with open(filename, 'r') as f:
        data = [float(i.strip('\n')) for i in f.readlines()]

    y = np.array(data)
    x = np.arange(y.shape[0])+1
    plt.figure(figsize=(15,10))
    plt.plot(x,y)
    plt.ylim(0,1)
    plt.title('test accuracy')
    plt.savefig('test_accuracy_chart.png')
    plt.clf()
    plt.close()
