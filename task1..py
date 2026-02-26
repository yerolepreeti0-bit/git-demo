# plots.py

import matplotlib.pyplot as plt

def sample_plot():
    x = [1, 2, 3, 4]
    y = [10, 20, 25, 30]
    
    plt.plot(x, y)
    plt.title("Sample Visualization")
    plt.show()

if __name__ == "__main__":
    sample_plot()