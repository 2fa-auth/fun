#!/home/client/Documents/fun/py/venv/bin/python3
import numpy as np

# правила BROADCASTING:
def normalize(x, mean, std): 
    mean = mean.reshape(1, 3, 1, 1) 
    std = std.reshape(1, 3, 1, 1) 
    return (x-mean) / std

def scale_channels(images, scale):
    scale = scale.reshape(1, 3, 1, 1)
    return images * scale


def main():
    x = np.arange(12).reshape(3, 4)
    mean = np.array([30, 40, 50, 60])
    std = np.array([0.2, 0.3, 0.1])
    normalize(x, mean, std)

    images = np.random.rand(16, 3, 64, 64)
    scale = np.array([0.5, 2.0, 1.5])
    scale_channels(images, scale)

    #axis:
    x = np.arange(24).reshape(2, 3, 4)
    print(x.shape)
    
    x.mean(axis=0)
    x.mean(axis=1)
    x.mean(axis=2)
    x.mean(axis=(1,2))

if __name__ == "__main__":
    main()    


