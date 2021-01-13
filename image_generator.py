import numpy as np
import matplotlib.pyplot as plt

def delta_func_img(coords, fluxes, extent, resolution):
    img = np.zeros([int((extent[3]-extent[2]) / resolution),
                    int((extent[1]-extent[0]) / resolution)])

    for i in range(len(fluxes)):

        adjusted_coords = [int((extent[2] - coords[i][1]) / resolution),
                           int((extent[0] - coords[i][0]) / resolution)]
#         print(adjusted_coords)
        try:
            img[adjusted_coords[0], adjusted_coords[1]] = fluxes[i]
        except IndexError:
            pass

    return img, resolution






if __name__ == '__main__':
    # Source 1:
    #       Position: J 05 00 00 +45 00 00
    #       Flux density: 3.6 Jy
    # Source 2:
    #       Position J 05 00 10 +45 03 00
    #       Flux density: 5.8 Jy
    star_coords = [[0, 0], [10, 180]]
    star_fluxes = [3.6, 5.8]

    extent = [-600, 600, -600, 600] # test FOV in arcseconds
    resolution = .1

    image, resolution = delta_func_img(star_coords, star_fluxes, extent, resolution, 'top left')
    extent_x = len(image[0])/2 * resolution
    extent_y = len(image)/2 * resolution
    plt.imshow(image, interpolation='gaussian', extent=[-extent_x, +extent_x, -extent_y, +extent_y])
    plt.title('original image')
    plt.colorbar()
    plt.show()

    image_fft = np.fft.fftshift(np.fft.ifft2(image))
    plt.imshow(np.abs(image_fft))
    plt.title('fourier transformed image')
    plt.colorbar()
    plt.show()
