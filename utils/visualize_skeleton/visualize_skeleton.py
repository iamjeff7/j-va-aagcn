import sys, os
import numpy as np
import data_info
from tqdm import tqdm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def prepare_data(sample):
    data = sample.reshape(300, -1, 3)
    data = list(data)

    # every data is padded to 300 frames
    # not useful in visualization
    current = data[0]
    stop_idx = 0
    for i in range(1, len(data)):
        if list(data[i][0]) == list(data[i][1]):
            stop_idx = i
            break
        else:
            current = data[i]

    # shape = (300, 50, 3)
    return data[:stop_idx]


def animate_scatters(iteration, data, scatters):
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters


def visualize(data, label, index, parent_folder='skeleton_videos', repeat=True, play=True, save=False):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initialize scatters
    scatters = []

    # Assign label and color for each joint
    # E.g. spine = black, left hand = red
    d = data[0]
    for i in range(d.shape[0]):
        i = i%25
        a = ax.scatter(d[i,0:1], 
                       d[i,1:2],
                       d[i,2:],
                       c=data_info.joint_color[data_info.ntu_joints[i]],
                       label=data_info.ntu_joints[i])
        scatters.append(a)

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    # if there are 2 actors, axes should be bigger
    if data[0].shape[0] == 25:
        ax.set_xlim3d([-1, 1])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1, 1])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-1, 1])
        ax.set_zlabel('Z')
    else:
        ax.set_xlim3d([-3, 3])
        ax.set_xlabel('X')

        ax.set_ylim3d([-3, 3])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-3, 3])
        ax.set_zlabel('Z')

    # Title of the graph
    fig.suptitle(data_info.classes[label])

    # Provide starting angle for the view.
    ax.view_init(-1, 1)

    ani = animation.FuncAnimation(fig, 
                                  animate_scatters,
                                  iterations,
                                  fargs=(data, scatters),
                                  interval=150,
                                  blit=False,
                                  repeat=repeat)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=50, metadata=dict(artist='Me'), 
                        extra_args=['-vcodec', 'libx264'])
        if not os.path.isdir(parent_folder):
            os.mkdir(parent_folder)
        index = index + '.mp4'
        filename = os.path.join(parent_folder, index)
        ani.save(filename, writer=writer, dpi=200)

    if play:
        plt.show()
    plt.clf()
    plt.close('all')

def main():
    # load skeleton
    x = 'sample_skeleton.npy'
    y = 'sample_label.npy'
    samples = np.load(x)
    labels = np.load(y)
    print('sample shape:', samples.shape)
    print('labels shape:', labels.shape)

    for i in tqdm(range(samples.shape[0])):
        if labels[i] < 49:
            sample = prepare_data(samples[i,:,:150//2])
        else:
            sample = prepare_data(samples[i])
        visualize(data=sample,
                  label=labels[i],
                  index=str(i),
                  play=False,
                  repeat=False,
                  save=True)


if __name__ == '__main__':
    print()
    main()
    print('\n\nDONE\n')

