import torch
from torch.distributions import MultivariateNormal

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

from system.Drone.base import BaseDrone


class DroneTrajectory(BaseDrone):
    """ Drone moving on an elliptical trajectory """

    def __init__(self, horizon, lambda_speed, lambda_nom, max_speed, wind_mu, wind_std, discrete_time, euler_time,
                 radius_x=5., radius_y=5., feasible_set=None, device="cpu"):
        super(DroneTrajectory, self).__init__(horizon=horizon, discrete_time=discrete_time,
                                              euler_time=euler_time, initial_radius=0., initial=(0., 0., 0.),
                                              feasible_set=feasible_set, device=device)

        # disturbance
        self.wind_mu = torch.tensor([wind_mu], device=self.device)
        self.wind_std = torch.tensor([wind_std], device=self.device)

        # ellipse axis
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.lambda_nom = lambda_nom

        # weight of the speed
        self.lambda_speed = lambda_speed
        self.max_speed = max_speed

    def initial_state(self, number_trajectories):
        """ samples "number_trajectories" initial states from P_0
         returns a tensor of shape ("number_trajectories", |S|) """
        # theta = torch.rand(number_trajectories, 1, device=self.device) * 2. * self.pi
        theta = torch.empty((number_trajectories, 1), device=self.device).fill_(-self.pi)
        xyz = torch.empty((number_trajectories, 3), device=self.device)
        xyz[..., (0,)] = self.radius_x * (1. + torch.cos(theta))
        xyz[..., (1,)] = self.radius_y * torch.sin(theta)
        xyz[..., (2,)] = 0.
        state_not_xyz = torch.zeros((number_trajectories, self.observation_space.shape[0] - 3), device=self.device)

        return torch.cat([state_not_xyz, xyz], dim=-1)

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t
            The faster, the closer we want to be to the trajectory such that the optimum is unique and stands for a zero
            deviation and an infinite speed.
            We thus penalize the distance to the closest point on the trajectory and reward the speed if the distance is
            within an exponential cone: distance < exp(- cte * speed) """
        distance_target = self._distance_target(states, actions, disturbances)
        speed_target = self._speed_target(states, actions, disturbances)
        distance_omega = self._distance_omega(states, actions, disturbances)

        # reward without the speed component
        reward = -distance_target - self.lambda_nom * distance_omega

        # add a reward depending on the speed
        speed_target_clip = torch.clamp_max(speed_target, max=self.max_speed)
        reward = reward + self.lambda_speed * speed_target_clip

        return reward

    def _distance_target(self, states, actions, disturbances):
        """ distance to the ellipse """
        _, _, _, xyz = states.split(3, dim=-1)
        x, y, z = xyz.split(1, dim=-1)
        x_ellipse, y_ellipse = self._closest_point_on_ellipse(x, y, z)

        distance = (x - x_ellipse).pow(2) + (y - y_ellipse).pow(2) + z.pow(2)

        return distance

    def _speed_target(self, states, actions, disturbances):
        """ radial speed """
        phi, theta, psi, p, q, r, u, v, w, x, y, z = states.split(1, dim=-1)

        x_ellipse, y_ellipse = self._closest_point_on_ellipse(x, y, z)

        # compute the angle
        theta_ellipse = torch.ones_like(x_ellipse) * self.pi / 2

        ind = x_ellipse != self.radius_x
        theta_ellipse[ind] = torch.arctan((y_ellipse[ind] * self.radius_x)
                                          / (torch.abs(x_ellipse[ind] - self.radius_x) * self.radius_y))

        ind = x_ellipse < self.radius_x
        theta_ellipse[ind] = self.pi - theta_ellipse[ind]

        # get the (opposite of the) derivative vector of the curve

        e_x = self.radius_x * torch.sin(theta_ellipse)
        e_y = -self.radius_y * torch.cos(theta_ellipse)
        norm = torch.sqrt(torch.pow(e_x, 2) + torch.pow(e_y, 2))
        e_x, e_y_ = e_x / norm, e_y / norm

        # get the direction in the local frame
        loc_dir_x = ((torch.cos(theta) * torch.cos(psi)) * e_x
                     + (torch.cos(theta) * torch.sin(psi)) * e_y)

        loc_dir_y = ((torch.sin(phi) * torch.sin(theta) * torch.cos(psi) - torch.cos(phi) * torch.sin(psi)) * e_x
                     + (torch.sin(phi) * torch.sin(theta) * torch.sin(psi) + torch.cos(phi) * torch.cos(psi)) * e_y)

        loc_dir_z = ((torch.cos(phi) * torch.sin(theta) * torch.cos(psi) + torch.sin(phi) * torch.sin(psi)) * e_x
                     + (torch.cos(phi) * torch.sin(theta) * torch.sin(psi) - torch.sin(phi) * torch.cos(psi)) * e_y)

        # get the speed component in that direction
        speed = loc_dir_x * u + loc_dir_y * v + loc_dir_z * w

        return speed

    def _closest_point_on_ellipse(self, x, y, z):
        y_ell = torch.empty_like(y)
        x_ell = torch.empty_like(x)

        ind = y != 0
        y_ell[ind] = torch.sign(y[ind]) * self.radius_x / torch.sqrt((self.radius_x / self.radius_y) ** 2
                                                                     + torch.pow((x[ind] - self.radius_x) / y[ind], 2))
        x_ell[ind] = (x[ind] - self.radius_x) * y_ell[ind] / y[ind] + self.radius_x

        ind = y == 0
        y_ell[ind] = 0.
        x_ell[ind] = self.radius_x * (1 + torch.sign(x[ind] - self.radius_x))

        return x_ell, y_ell

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return WindDist(self.wind_mu, self.wind_std, self.device, states)

    def to_gym(self):
        raise NotImplementedError

    def render(self, states, actions, dist, rewards, num_trj):
        # return self._render_dynamical(states, actions, dist, rewards, num_trj)
        return self._render_plot(states, actions, dist, rewards, num_trj)

    def _render_plot(self, states, actions, dist, rewards, num_trj):
        """ fixed graphical view """
        x, y, z, x_, y_, z_, dist_target = self._render_process(states, actions, dist)

        # crate the figure
        fig = plt.figure()
        ax_trajectory = fig.add_subplot(1, 1, 1, projection='3d')
        animation_f = self._animate(ax_trajectory, [], [], states, actions, dist, rewards)
        animation_f(0, x.shape[1] - 1)
        plt.tight_layout()
        plt.savefig('drone-desga.pdf')
        plt.show()

    def _render_dynamical(self, states, actions, dist, rewards, num_trj):
        """ animated graphical view """
        x, y, z, x_, y_, z_, dist_target = self._render_process(states, actions, dist)

        # crate the figure
        fig = plt.figure()
        nb_figures = 4
        tj_id_buffer = [0]

        # 3d trajectory plot
        ax_trajectory = fig.add_subplot(1, 2, 1, projection='3d')

        # 2d time plot
        axs_time = [fig.add_subplot(3, 2, 2*i) for i in range(1, nb_figures)]

        animation_f = self._animate(ax_trajectory, axs_time, tj_id_buffer, states, actions, dist, rewards)

        def animate(i):
            ax_trajectory.cla()
            for ax in axs_time:
                ax.cla()

            """ select the trajectory """
            tj_id = tj_id_buffer.pop()

            if i >= x.shape[1]-1:
                tj_id_buffer.append((tj_id + 1) % x.shape[0])
            else:
                tj_id_buffer.append(tj_id)

            animation_f(tj_id, i)

        ani = FuncAnimation(fig, animate, x.shape[1])
        # ani.save('drone.gif')
        plt.tight_layout()
        plt.show()

    def _render_process(self, states, actions, dist):
        nb_digits = 3
        _, _, _, xyz = states[:, :-1, :].split(3, dim=-1)
        x, y, z = (torch.round(xyz * 10**nb_digits) / (10**nb_digits)).split(1, dim=-1)

        _, _, _, dot_xyz = self._derivative(states[:, :-1, :], actions, dist).split(3, dim=-1)
        dot_x, dot_y, dot_z = (torch.round(dot_xyz * 10 ** nb_digits) / (10 ** nb_digits)).split(1, dim=-1)
        x_, y_, z_ = x + self.discrete_time * dot_x, y + self.discrete_time * dot_y, z + self.discrete_time * dot_z

        dist_target = self._distance_target(states, actions, dist)

        control_perf = torch.cat([-dist_target[:, :-1, :],
                                  self.lambda_speed * self._speed_target(states[:, :-1, :], actions, dist)], dim=-1)

        return x, y, z, x_, y_, z_, control_perf

    def _animate(self, ax_trajectory, axs_time, tj_id_buffer, states, actions, dist, rewards):
        # compute the ellipse points
        theta = torch.linspace(0, 2 * 3.14159, 200)
        x_ellipse = self.radius_x * (1 + torch.cos(theta))
        y_ellipse = self.radius_y * torch.sin(theta)
        z_ellipse = torch.zeros_like(x_ellipse)
        x_ellipse, y_ellipse, z_ellipse = x_ellipse.numpy(), y_ellipse.numpy(), z_ellipse.numpy()

        _, _, _, _, _, _, _, _, _, x, y, z = states.split(1, dim=-1)
        px_ellipse, py_ellipse = self._closest_point_on_ellipse(x, y, z)
        pz_ellipse = torch.zeros_like(px_ellipse)
        px_ellipse, py_ellipse, pz_ellipse = px_ellipse.numpy(), py_ellipse.numpy(), pz_ellipse.numpy()

        """ do the preprocessing """
        x, y, z, x_, y_, z_, control_perf = [t.numpy() for t in self._render_process(states, actions, dist)]

        """ printing function """
        def animation_f(tj_id, i):
            """ ellipse """
            ax_trajectory.plot(x_ellipse, y_ellipse, z_ellipse, 'green')
            # ax_trajectory.scatter(px_ellipse[tj_id, i, 0], py_ellipse[tj_id, i, 0], pz_ellipse[tj_id, i, 0], color='k')

            """ 3d plots """
            ax_trajectory.set_xlabel('x')
            ax_trajectory.set_ylabel('y')
            ax_trajectory.set_zlabel('z')

            ax_trajectory.scatter(x[tj_id, :i+1, 0], y[tj_id, :i+1, 0], -z[tj_id, :i+1, 0], color='b')
            ax_trajectory.set_zlim(-1., 1.)

            """
            ax_trajectory.plot(x[tj_id, :i+1, 0], z[tj_id, :i+1, 0], 'c+', zdir='y', zs=1.5)
            ax_trajectory.plot(y[tj_id, :i+1, 0], z[tj_id, :i+1, 0], 'm+', zdir='x', zs=-0.5)
            ax_trajectory.plot(x[tj_id, :i+1, 0], y[tj_id, :i+1, 0], 'y+', zdir='z', zs=-1.5)

            arr = Arrow3D(x[tj_id, i, 0], y[tj_id, i, 0], -z[tj_id, i, 0],
                          x_[tj_id, i, 0], y_[tj_id, i, 0], -z_[tj_id, i, 0],
                          mutation_scale=20,
                          arrowstyle="-|>")
            ax_trajectory.add_artist(arr)
            """

            for set_lim, get_lim, arr, arr_ in zip([ax_trajectory.set_xlim, ax_trajectory.set_ylim,
                                                    ax_trajectory.set_zlim],
                                                   [ax_trajectory.get_xbound, ax_trajectory.get_ybound,
                                                    ax_trajectory.get_zbound],
                                                   [x, y, -z],
                                                   [x_, y_, -z_]):
                min_val, max_val = get_lim()
                min_val = min(arr[tj_id, :i+1, 0].min(), arr_[tj_id, i, 0], min_val)-0.01
                max_val = max(arr[tj_id, :i+1, 0].max(), arr_[tj_id, i, 0], max_val)
                scale = 0.15 * (max_val - min_val)
                set_lim(min_val - scale, max_val + scale)

            """ 2d plots """
            for ax, value, titles in zip(axs_time,
                                         [rewards[tj_id, :i+1, :], actions[tj_id, :i+1, :],
                                          control_perf[tj_id, :i+1, :]],
                                         ['Rewards', "Actions", "Control performance"]):
                ax.set_title(titles)
                ax.plot(value)
                ax.set_xlim(0, x.shape[1])

        return animation_f

    def control_perf(self, states, actions, disturbances, rewards):
        """Evaluates the performance of the controller only"""
        return self._distance_target(states, actions, disturbances).mean()


class WindDist:

    def __init__(self, loc, scale, device, states, *args, **kwargs):
        super(WindDist, self).__init__(*args, **kwargs)
        self.device = device

        loc_vect = loc.view(-1, 3)
        covariance_matrix = torch.diag_embed(scale.view(-1, 3).pow(2) + 10.0e-15)

        self.dist = MultivariateNormal(loc=loc_vect.repeat_interleave(states.shape[0], dim=0),
                                       covariance_matrix=(covariance_matrix.repeat_interleave(states.shape[0], dim=0)))

    def sample(self):
        loc_f = self.dist.sample()
        loc_tau = torch.zeros_like(loc_f)

        return torch.cat([loc_f, loc_tau], dim=-1).to(self.device)

    def log_prob(self, value):
        f, _ = value.split(3, dim=-1)

        return self.dist.log_prob(f).sum(dim=-1, keepdims=True)


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
