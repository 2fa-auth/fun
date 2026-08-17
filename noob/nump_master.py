"""
немного поиграться с основными правилами тензоров:
1. broadcasting (и его правилами)
2. axis
3. inicing
4. vectorization 
"""


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
    mean = np.array([0.5, 0.4, 0.3])
    std = np.array([0.2, 0.3, 0.1])
    normalize(x, mean, std)

    images = np.random.rand(16, 3, 64, 64)
    scale = np.array([0.5, 2.0, 1.5])
    scale_channels(images, scale)

    #axis:
    x = np.arange(24).reshape(2, 3, 4)
    x.mean(axis=0)
    x.mean(axis=1)
    x.mean(axis=2)
    x.mean(axis=(1,2))

    #indicing
    x = np.arange(24).reshape(2, 3, 4)
    print(x)
    print()
    print(x[0].shape) # (2, 3, 4) - > (3, 4)
    print(x[:, 0].shape) # (2, 3, 4) - > (2, 4)
    print(x[:, :, 0].shape) # (2, 3, 4) - > (2, 3)

    print(x[0, :, :].shape)
    print(x[0, 1, :].shape) 
    print(x[:, 1:3, :].shape)
    print(x[:, :, 1:3].shape)
    print(x[0:2, 0:2, 0:2].shape) 
    print(x[..., 0].shape)
    print(x[1, 2, 3].shape)

    #axis + indicing + broadcasting
    images = np.random.rand(16, 3, 64, 64)
    # где:
    #   16 - кол-во изображений
    #   3 - кол-во каналов
    #   64x64 - кол-во пикселей (размер изобр) 
    brightness = np.array([0.8, 1.0, 1.2])
 
    # умножение каждого канала на соответствующий коэфициент:
    brightness = brightness.reshape(1, 3, 1, 1)
    print((images * brightness))


    #vectorization
    

if __name__ == "__main__":
    main()    



