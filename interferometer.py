import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import image_generator
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D

def diameter_to_ang_res(diameter, wavelength = (299792458)/(5*10**9)):
#     wavelength from 5GHz observing freq
#     wavelength = 299,792,458/freq(Hz)
#     returns angular resolution in arcseconds
    ang_res = 1.22*wavelength/diameter * 180 / np.pi * 3600
    return ang_res

def choose_ant(source='telescope_positions.csv', configuration="D"):
    # choose the correct antenna configuration from csv file which shows antenna positions
    # link to pdf from which the csv is made
    # https://science.nrao.edu/facilities/vla/docs/manuals/oss2016A/ant_positions.pdf
    df = pd.read_csv(source, true_values='X', false_values='-')
    config_filter = df[configuration]==True
    configured = df.where(config_filter).dropna().drop(['A', 'B','C', 'D'],
                                                       axis=1).reset_index(drop=True)
    return configured


def convert_XYZ_to_UVW(pos, dec, ha):
    pos_matrix = pos.loc[:, ['Lx(ns)', 'Ly(ns)', 'Lz(ns)']].to_numpy().transpose()*0.3
    rot_matrix = np.array([[np.sin(ha),                 np.cos(ha),                 0],
                           [-np.sin(dec)*np.cos(ha),    np.sin(dec)*np.sin(ha),     np.cos(dec)],
                           [np.cos(dec)*np.cos(ha),     -np.cos(dec)*np.sin(ha),    np.sin(dec)]])
    UVW_matrix = rot_matrix.dot(pos_matrix) * 299792458 * 10**(-9)
    return UVW_matrix


def make_baselines(declination, hour_angles, positions):
    full_baseline = [[], [], []]

    for ha in hour_angles:
        # each loop makes full baseline set for each HA
        
        UVW = convert_XYZ_to_UVW(positions, declination, ha)
        no_of_baselines = len(UVW[0]) * (len(UVW[0]) - 1)
        # number of baselines equals N(N-1), where N is number of telescopes
        # as we count both the positive and negative vectors
        
        baselines = np.array([[0.] * no_of_baselines] * 3)

        index = 0
        for i in range(len(UVW[0])):
            for j in range(len(UVW[0])):
                if i == j:
                    continue
                baselines[0, index] = UVW[0, i] - UVW[0, j]
                baselines[1, index] = UVW[1, i] - UVW[1, j]
                baselines[2, index] = UVW[2, i] - UVW[2, j]
                index += 1

        full_baseline[0].extend(baselines[0])
        full_baseline[1].extend(baselines[1])
        full_baseline[2].extend(baselines[2])
    return np.array(full_baseline)




def roundup(x, val):
    return int( np.ceil( x / val )) * val


def make_sampling_func(src, res = 1):
    # resolution (res) is in metres

    # arbitrary limits, chosen to look nice
    x_max = int(np.ceil(350 / res))
    x_min = int(np.floor(-350 / res))
    y_max = int(np.ceil(350 / res))
    y_min = int(np.floor(-350 / res))

    sampling_func = np.zeros([y_max-y_min, x_max-x_min])

    for i in range(len(src[0])):
        # increasing from top left of array
        x = int(src[0, i]/res - x_min)
        y = int(src[1, i]/res-y_min)
        try:
            sampling_func[y,x] = 1
        except (IndexError):
            pass
    return sampling_func, res


def fft_with_scaling(function, resolution, wavelength=0.05995849):
    # wavelength has same reasoning as diameter_to_ang_res function
    transed_func = np.fft.fftshift(np.fft.fft2(function))
    new_res = diameter_to_ang_res(resolution*len(function), wavelength)
    # largest possible distance in old -> smallest possible distance in new
    return transed_func, new_res


if __name__ == '__main__':
    # choose the correct antenna configuration
    positions = choose_ant()
    print(positions)

    # declination and list of observed hour_angles
    declination = 45
    hour_angles = np.arange(-0.5, 0.5, 30/3600)
    baseline = make_baselines(declination, hour_angles, positions)
    print(baseline)
    full = plt.figure()
    full_ax = full.add_subplot(111)
    full_ax.set_title('baselines')
    full_ax.scatter(baseline[0], baseline[1])
    plt.show()

    
    # testing out the resolution function
    res = 1
    sampling_function, res = make_sampling_func(baseline, res)
    print('sampling_function', sampling_function.shape)
    extent_x = len(sampling_function[0])/2 * res
    extent_y = len(sampling_function)/2 * res
    plt.imshow(sampling_function, interpolation='gaussian',
               extent=[-extent_x, +extent_x, -extent_y, +extent_y])
    plt.title('sampling function')
    plt.colorbar()
    plt.show()

    res = 0.1
    sampling_function, res = make_sampling_func(baseline, res)
    print('sampling_function', sampling_function.shape)
    extent_x = len(sampling_function[0])/2 * res
    extent_y = len(sampling_function)/2 * res
    plt.imshow(sampling_function, interpolation='gaussian', extent=[-extent_x, +extent_x, -extent_y, +extent_y])
    plt.title('sampling function')
    plt.colorbar()
    plt.show()

    # testing the dirty beam
    dirty_beam = np.fft.fftshift(np.fft.fft2(sampling_function))
    plt.imshow(np.abs(dirty_beam))
    plt.title('dirty beam')
    plt.colorbar()
    plt.show()

    transed, transed_res = fft_with_scaling(sampling_function, res)
    extent_x = len(transed[0])/2 * transed_res
    extent_y = len(transed)/2 * transed_res
    plt.imshow(np.abs(dirty_beam), extent=[-extent_x, +extent_x, -extent_y, +extent_y])
    plt.title('transed beam')
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(dirty_beam), norm=colors.LogNorm())
    plt.title('dirty beam')
    plt.colorbar()
    plt.show()

    true_image = image_generator.true_image()
    true_image_response = np.fft.fftshift(np.fft.ifft2(true_image))
    plt.imshow(np.abs(true_image_response), interpolation='gaussian')
    plt.title('true image response')
    plt.colorbar()
    plt.show()

    print('true_image_response', true_image_response.shape)
    print('dirty_beam', dirty_beam.shape)


    # testing convolution
    
    
    # print('starting signal.oaconvolve')
    # t = time.time()
    # actual_response = signal.oaconvolve(true_image_response, dirty_beam)
    # t = time.time()-t
    # print('finished signal.oaconvolve: time taken =', t)
    # plt.imshow(np.abs(actual_response), interpolation='gaussian')
    # plt.title('actual response')
    # plt.colorbar()
    # plt.show()

    print('starting signal.convolve')
    t = time.time()
    actual_response = signal.convolve(true_image_response, dirty_beam)
    t = time.time()-t
    print('finished signal.convolve: time taken =', t)
    plt.imshow(np.abs(actual_response), interpolation='gaussian')
    plt.title('actual response')
    plt.colorbar()
    plt.show()
