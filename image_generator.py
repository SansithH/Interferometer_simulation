import numpy as np
import matplotlib.pyplot as plt

def delta_func_img(coords, fluxes, extent, zero_pos = 'centre'):
    if 'centre' in zero_pos:
        y_adjust = int(extent[0] / 2)
        x_adjust = int(extent[1] / 2)
    if 'bottom' in zero_pos:
        y_adjust = extent[0]-1
    elif 'top' in zero_pos:
        y_adjust = 0
    if 'right' in zero_pos:
        x_adjust = extent[1]-1
    elif 'left' in zero_pos:
        x_adjust = 0
    # print(y_adjust, x_adjust)

    img = np.zeros(extent)
    # increasing from top right
    for i in range(len(fluxes)):
        adjusted_coords = [coords[i][1]+y_adjust, coords[i][0]+x_adjust]
        print(adjusted_coords)
        try:
            img[adjusted_coords[0], adjusted_coords[1]] = fluxes[i]
        except IndexError:
            pass

    return img


def true_image():
    star_coords = [[0, 0], [10, 180]]
    star_fluxes = [3.6, 5.8]
    extent = [2000, 2200]

    image = delta_func_img(star_coords, star_fluxes, extent, 'centre')

    return image





if __name__ == '__main__':
    # Source 1:
    #       Position: J 05 00 00 +45 00 00
    #       Flux density: 3.6 Jy
    # Source 2:
    #       Position J 05 00 10 +45 03 00
    #       Flux density: 5.8 Jy
    star_coords = [[0, 0], [10, 180]]
    star_fluxes = [3.6, 5.8]

    extent = [400, 400]

    image = delta_func_img(star_coords, star_fluxes, extent, 'centre')
    plt.imshow(image)
    plt.title('original image')
    plt.colorbar()
    plt.show()

    image_fft = np.fft.fftshift(np.fft.ifft2(image))
    plt.imshow(np.abs(image_fft))
    plt.title('fourier transformed image')
    plt.colorbar()
    plt.show()
