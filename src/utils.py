import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

'''
    report: i           iteration number.
            x_i         location xi.
            f_x_i       objective value f(xi).
            step_len    step length taken ‖𝑥𝑖 − 𝑥𝑖−1‖
            obj_change  change in objective function value |𝑓(𝑥𝑖) − 𝑓(𝑥𝑖−1)|.
'''
def iteration_reporting(i, x_i, f_x_i, step_len, obj_change):
    print("Iteration number = ", i, "\n" \
          "\t", "Location = ", x_i, "\n" \
          "\t", "Objective value = ", f_x_i, "\n" \
          "\t", "Step length taken = ", step_len, "\n" \
          "\t", "Change in objective function = ", obj_change);


def plot_outlines(f, sol, start_point, history, label):
    x = np.linspace(sol[0][0] - start_point[0][0], sol[0][0] + start_point[0][0], 20)
    y = np.linspace(sol[1][0] - start_point[1][0], sol[1][0] + start_point[1][0], 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            vec = np.array([[x[i]], [y[j]]])
            f_x, df_x, hess_x = f(vec)
            Z[i][j] = f_x

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z.T)

    ax.plot(history[-1]['x_next'][0][0], history[-1]['x_next'][1][0], 'o')
    for k in range(len(history)):
        ax.annotate('', xy=history[k]['x_next'], xytext=(history[k]['x_prev']*1.05),
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 0.8}, va='center', ha='center')

    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(label)

    fig.savefig('../tests/plots/'+label+'.png')
    plt.close(fig)

def plot_objective_vals(history, label):
    x = []
    y = []

    plt.rcParams["figure.figsize"] = (15, 13)
    fig, ax = plt.subplots()

    for k in range(len(history)):
        x.append(history[k]['iter'])
        y.append(float(history[k]['f_next']))

    # Plot the data
    plt.plot(x, y, label=label)
    plt.xlabel('Number of iteration', fontsize=18)
    plt.ylabel('Objective function value', fontsize=16)
    fig.savefig('../tests/plots/' + label + '.png')
    plt.close(fig)


def plot_3d_outlines(f, sol, start_point, history, label):
    x = np.linspace(sol[0][0] - start_point[0][0], sol[0][0] + start_point[0][0], 20)
    y = np.linspace(sol[1][0] - start_point[1][0], sol[1][0] + start_point[1][0], 20)
    z = np.linspace(sol[2][0] - start_point[2][0], sol[2][0] + start_point[2][0], 20)
    # X, Y, Z = np.meshgrid(x, y, z)
    # c = X**X + Y**Y + (Z+1)**(Z+1)
    c = np.zeros((len(x), len(y), len(z)))

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                vec = np.array([[x[i]], [y[j]], [z[k]]])
                f_x, df_x, hess_x = f(vec)
                c[i][j][k] = f_x

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = c > 0.01
    idx = np.arange(int(np.prod(c.shape)))
    x, y, z = np.unravel_index(idx, c.shape)
    ax.scatter(x, y, z, c=c.flatten(), s=10 * mask, edgecolor="face", alpha=0.8, cmap="plasma", linewidth=0)

    ax.set_xlim(sol[0][0] - start_point[0][0], sol[0][0] + start_point[0][0])
    ax.set_ylim(sol[1][0] - start_point[1][0], sol[1][0] + start_point[1][0])
    ax.set_zlim(sol[2][0] - start_point[2][0], sol[2][0] + start_point[2][0])

    ax.scatter(history[-1]['x_next'][0][0], history[-1]['x_next'][1][0], history[-1]['x_next'][2][0], 'o')
    for k in range(1, len(history)):
        a = Arrow3D([history[k]['x_prev'][0][0], history[k]['x_next'][0][0]],
                    [history[k]['x_prev'][1][0], history[k]['x_next'][1][0]],
                    [history[k]['x_prev'][2][0], history[k]['x_next'][2][0]],
                    mutation_scale=10,
                    lw=1, arrowstyle="->", color="r")
        ax.add_artist(a)

    plt.tight_layout()
    fig.savefig('../tests/plots/'+label+'.png')
    plt.close(fig)


def plot_linear_feasible_region(label):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    d = np.linspace(-1, 4, 2000)
    x, y = np.meshgrid(d, d)
    ax.imshow(((y >= 0) & (y >= 1 - x) & (y <= 1) & (x <= 2)).astype(int),
               extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greys", alpha=0.3);

    # Make plot
    plt.plot(d, np.ones_like(d), label=r'$y\leq1$')
    plt.plot(d, np.zeros_like(d), label=r'$y\geq0$')
    plt.plot(d, (((-1) * d) + 1), label=r'$y\geq -x + 1$')
    plt.plot(2*np.ones_like(d), d, label=r'$x\leq2$')
    plt.xlim((-1, 4))
    plt.ylim((-1, 4))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # Fill feasible region
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.savefig('../tests/plots/' + label + '.png')
    plt.close(fig)

def plot_feasible_region_3d(label):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # plot the plane
    xx, yy = np.meshgrid(range(10), range(10))
    zz = np.zeros_like(xx)

    ax.plot_surface(xx, yy, zz, alpha=0.5, label=r'$z\geq0$')
    ax.plot_surface(xx, zz, yy, alpha=0.5, label=r'$y\geq0$')
    ax.plot_surface(zz, yy, xx, alpha=0.5, label=r'$x\geq0$')
    ax.plot_surface(xx, yy, 1-xx-yy, alpha=0.5, label=r'$x+y+z=1$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # # Fill feasible region
    fig.savefig('../tests/plots/' + label + '.png')
    plt.close(fig)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)