import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import image_generator
from scipy import signal
import time
import pydeconv
from mpl_toolkits.mplot3d import Axes3D

def choose_ant(source='telescope_positions.csv', configuration="D"):
    # choose the correct antenna configuration
    df = pd.read_csv(source, true_values='X', false_values='-')
    config_filter = df[configuration]==True
    configured = df.where(config_filter).dropna().drop(['A', 'B','C', 'D'], axis=1).reset_index(drop=True)
    return configured


def convert_XYZ_to_UVW(pos, dec, ha):
    pos_matrix = pos.loc[:, ['Lx(ns)', 'Ly(ns)', 'Lz(ns)']].to_numpy().transpose()*0.3
    rot_matrix = np.array([[np.sin(ha),                 np.cos(ha),                 0],
                           [-np.sin(dec)*np.cos(ha),    np.sin(dec)*np.sin(ha),     np.cos(dec)],
                           [np.cos(dec)*np.cos(ha),     -np.cos(dec)*np.sin(ha),    np.sin(dec)]])
    UVW_matrix = rot_matrix.dot(pos_matrix)
    return UVW_matrix


def make_baselines(declination, hour_angles, positions):
    full_baseline = [[], [], []]

    for ha in hour_angles:
        UVW = convert_XYZ_to_UVW(positions, declination, ha)
        no_of_baselines = len(UVW[0]) * (len(UVW[0]) - 1)
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


def make_sampling_func(src):
    print(src)
    print(src.shape)
    round_to = 100
    x_max = int(np.ceil(max(src[0]) / round_to)) * round_to
    x_min = int(np.floor(min(src[0]) / round_to)) * round_to
    y_max = int(np.ceil(max(src[1]) / round_to)) * round_to
    y_min = int(np.floor(min(src[1]) / round_to)) * round_to
    z_max = int(np.ceil(max(src[2]) / round_to)) * round_to
    z_min = int(np.floor(min(src[2]) / round_to)) * round_to

    sampling_func = np.zeros([y_max-y_min, x_max-x_min])

    for i in range(len(src[0])):
        # increasing from top left of array
        x = int(src[0, i] - x_min)
        y = int(src[1, i]-y_min)

        sampling_func[y,x] = 1

    return sampling_func



if __name__ == '__main__':
    # choose the correct antenna configuration
    positions = choose_ant()
    # print(positions)

    # declination and list of observed hour_angles
    declination = 45
    hour_angles = np.arange(-0.5, 0.5, 30/3600)
    baseline = make_baselines(declination, hour_angles, positions)
    # print(baseline)
    # full = plt.figure()
    # full_ax = full.add_subplot(111)
    # full_ax.set_title('baselines')
    # full_ax.scatter(baseline[0], baseline[1])
    # plt.show()

    sampling_function = make_sampling_func(baseline)
    print('sampling_function', sampling_function.shape)
    # plt.imshow(sampling_function, interpolation='gaussian')
    # plt.title('sampling function')
    # plt.colorbar()
    # plt.show()

    dirty_beam = np.fft.fftshift(np.fft.fft2(sampling_function))
    plt.imshow(np.abs(dirty_beam))
    plt.title('dirty beam')
    plt.colorbar()
    plt.show()

    # width = 200
    # height = 200
    # offset_x = dirty_beam.shape[1]//2
    # offset_y = dirty_beam.shape[0]//2
    #
    # cut_dirty_beam = dirty_beam[offset_y-height//2:offset_y+height//2, offset_x-width//2:offset_x+width//2]
    # plt.imshow(np.abs(cut_dirty_beam))
    # plt.title('cut_dirty beam')
    # plt.colorbar()
    # plt.show()

    true_image = image_generator.true_image()
    true_image_response = np.fft.fftshift(np.fft.ifft2(true_image))
    plt.imshow(np.abs(true_image_response), interpolation='gaussian')
    plt.title('true image response')
    plt.colorbar()
    plt.show()

    print('true_image_response', true_image_response.shape)
    print('dirty_beam', dirty_beam.shape)


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

    # manual convolve
    star_coords = [[0, 0], [10, 180]]
    star_fluxes = [3.6, 5.8]
    extent = [2000, 2200]


    # deconvolve

    reconstructed_response, noise_map = pydeconv.hogbom(np.abs(actual_response), np.abs(dirty_beam), True, 0.2, 0.3, 5)
    plt.imshow(np.abs(noise_map), interpolation='gaussian')
    plt.title('noise_map')
    plt.colorbar()
    plt.show()
    plt.imshow(np.abs(reconstructed_response), interpolation='gaussian')
    plt.title('reconstructed_response')
    plt.colorbar()
    plt.show()











    #
    # # original = plt.figure()
    # # orig_ax = original.add_subplot(111)
    # # orig_ax.scatter(positions["Lx(ns)"], positions["Ly(ns)"])
    # # orig_ax.set_title('Original positions')
    # # orig_ax.set_xlabel("Lx(ns)")
    # # orig_ax.set_ylabel("Ly(ns)")
    # # plt.show()
    #
    #
    # declination = 45.0
    # hour_angles = np.arange(-0.5, 0.5, 30/3600)
    # full_baseline = [[],[],[]]
    #
    # # base = plt.figure()
    # # base.canvas.manager.full_screen_toggle()
    # # base_ax = base.add_subplot(121)
    # # base_ax.set_xbound(-1100, 1100)
    # # curr = base.add_subplot(122)
    #
    #
    #
    # for ha in hour_angles:
    #     UVW = convert_XYZ_to_UVW(positions, declination, ha)
    #     # print(UVW.shape)
    #
    #     no_of_baselines = len(UVW[0])**2-len(UVW[0])
    #     baselines = np.array([[0.]*no_of_baselines]*3)
    #     index = 0
    #     for i in range(len(UVW[0])):
    #         for j in range(len(UVW[0])):
    #             if i == j:
    #                 continue
    #             baselines[0, index] = UVW[0, i] - UVW[0, j]
    #             baselines[1, index] = UVW[1, i] - UVW[1, j]
    #             baselines[2, index] = UVW[2, i] - UVW[2, j]
    #             index += 1
    #
    #     # print(baselines.shape)
    #
    #     full_baseline[0].extend(baselines[0])
    #     full_baseline[1].extend(baselines[1])
    #     full_baseline[2].extend(baselines[2])
    #
    #
    #     # ha_text = 'HA = {:.3f}'.format(ha)
    #     #
    #     # curr.cla()
    #     # curr.scatter(UVW[0], UVW[1], c='r')
    #     # curr.set_title('Current UV positions with {}'.format(ha_text))
    #     # curr.set_xbound(-700, 700)
    #     # curr.set_ybound(-700, 700)
    #     #
    #     # base_ax.scatter(baselines[0], baselines[1], c='blue')
    #     # base_ax.set_title('Cumulative baselines at {}'.format(ha_text))
    #     # plt.show(block=False)
    #     # plt.pause(0.01)
    #
    #
    #
    #
    #
    #
    # # print(np.array(full_baseline).shape)
    # # full = plt.figure()
    # # full_ax = full.add_subplot(111)
    # # full_ax.scatter(full_baseline[0], full_baseline[1])
    # # plt.show()
