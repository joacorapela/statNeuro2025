"""Utils (mostly plotting) for NeuroRL tutorial"""

from typing import Any
import numpy as np
import xarray as xr 
from tqdm import tqdm
import matplotlib.pyplot as plt
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
ratinabox.stylize_plots() # sets some RC params to make plots look better
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.xmargin'] = 0.05
rcParams['axes.ymargin'] = 0.05
rcParams['legend.frameon'] = True
rcParams['legend.scatterpoints'] = 4
rcParams["legend.framealpha"] = 0.1
rcParams["legend.edgecolor"] = [1,1,1,0]
rcParams["text.usetex"] = False

small_action_dict = {
    0: {"name":"North", "label":"↑", "delta":(0,1)},
    1: {"name":"East", "label":"→", "delta":(1,0)},
    2: {"name":"South", "label":"↓", "delta":(0,-1)},
    3: {"name":"West", "label":"←", "delta":(-1,0)},}

large_action_dict = {
    0: {"name":"North", "label":"↑", "delta":(0,1)},
    1: {"name":"North-East", "label":"↗", "delta":(np.sqrt(2)/2,np.sqrt(2)/2)},
    2: {"name":"East", "label":"→", "delta":(1,0)},
    3: {"name":"South-East", "label":"↘", "delta":(np.sqrt(2)/2,-np.sqrt(2)/2)},
    4: {"name":"South", "label":"↓", "delta":(0,-1)},
    5: {"name":"South-West", "label":"↙", "delta":(-np.sqrt(2)/2,-np.sqrt(2)/2)},
    6: {"name":"West", "label":"←", "delta":(-1,0)},
    7: {"name":"North-West", "label":"↖", "delta":(-np.sqrt(2)/2,np.sqrt(2)/2)},}

def format_axes(ax, xlims=None, ylims=None):
    """
    Format the axes by removing top and right spines, setting the bounds of the left and bottom spines,
    and setting the x and y limits if provided.
    
    Parameters:
    - ax: The axes object to format.
    - xlims: The x-axis limits as a tuple (min, max). If None, the limits are determined from the data.
    - ylims: The y-axis limits as a tuple (min, max). If None, the limits are determined from the data.
    
    Returns:
    - The formatted axes object.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xlims = (xlims or (ax.dataLim.bounds[0], ax.dataLim.bounds[2]))
    ylims = (ylims or (ax.dataLim.bounds[1], ax.dataLim.bounds[3]))
    ax.spines['left'].set_bounds(ylims[0], ylims[1])
    ax.spines['bottom'].set_bounds(xlims[0], xlims[1])
    return ax

class BaseRescorlaWagner:
    """
    Base class for the Rescorla-Wagner learning algorithm.
    """
    def __init__(self, n_stimuli=1, alpha=0.1):
        """
        Initialize the Rescorla-Wagner learning algorithm.
        
        Parameters:
        - n_stimuli: The number of stimuli.
        - alpha: The learning rate.
        """
        self.n_stimuli = n_stimuli
        if n_stimuli == 1:
            self.V_history = [self.V]
        elif n_stimuli > 1: 
            self.V_history = [0]
            self.W_history = [self.W.copy()]
            self.S_history = [np.zeros(n_stimuli)]
            self.stim_names = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        self.alpha = alpha
        self.R_history = [0]

    def plot(self, ax=None):
        """
        Plot the predicted value and reward history.
        
        Parameters:
        - ax: The axes object to plot on. If None, a new figure and axes will be created.
        
        Returns:
        - The axes object.
        """
        if self.n_stimuli == 1:
            ax = self.plot_1d(ax)
        else:
            ax = self.plot_2d(ax)
        return ax 
        
    def plot_1d(self, ax=None):
        """
        Plot the predicted value and reward history in 1D.
        
        Parameters:
        - ax: The axes object to plot on. If None, a new figure and axes will be created.
        
        Returns:
        - The axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2)) 
        ax.scatter(np.arange(len(self.V_history)), self.V_history, label='Predicted value', color='C0', linewidth=0, s=5, alpha=0.7) 
        ax.scatter(np.arange(len(self.R_history)), self.R_history, label='Reward', color='orange', linewidth=0, alpha=0.7)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Value')
        ax.legend()
        ax = format_axes(ax)
        return ax
    
    def plot_2d(self, ax=None):
        """
        Plot the predicted value and reward history in 2D.
        
        Parameters:
        - ax: The axes object to plot on. If None, a new figure and axes will be created.
        
        Returns:
        - The axes objects.
        """
        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 4, 2])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        N_trial = len(self.V_history)
        S_history = np.array(self.S_history); maxS, minS = np.max(S_history), np.min(S_history)
        R_history = np.array(self.R_history); maxR, minR = np.max(R_history), np.min(R_history)   
        W_history = np.array(self.W_history); maxW, minW = np.max(W_history), np.min(W_history)
        V_history = np.array(self.V_history); maxV, minV = np.max(V_history), np.min(V_history)

        for t in range(N_trial):
            for i in range(self.n_stimuli):
                ax1.add_artist(plt.Rectangle((t-0.5, i-0.5), 1, 1, color=f'C{i+1}', alpha = (S_history[t,i] - minS) / max(1, maxS), linewidth=0))
                ax2.scatter(t, W_history[t,i], label=f'Stimulus {self.stim_names[i]} weight' if t == 0 else None, color=f'C{i+1}', linewidth=0, s=8, alpha=0.7)
            ax3.scatter(t, V_history[t], label='Predicted (s.w)' if t==0 else None, s=8, color='C0', linewidth=0, alpha=0.7)
            ax3.scatter(t, R_history[t], label='Actual (reward recieved)' if t==0 else None, color='orange', linewidth=0, alpha=0.7)
        
        # format axes 
        ax1 = format_axes(ax1, ylims=(0, self.n_stimuli-1), xlims=(0, N_trial-1))
        ax1.set_yticks(np.arange(self.n_stimuli))
        ax1.set_yticklabels([f'{self.stim_names[i]}' for i in range(self.n_stimuli)])
        ax1.set_ylim(-0.5, self.n_stimuli-0.5)
        ax2 = format_axes(ax2)
        ax3 = format_axes(ax3)

        # y axis spines 
        # ax1.spines['left'].set_visible(False)
        ax1.set_yticks(np.arange(self.n_stimuli))

        # labels and legends
        ax1.set_ylabel('Stimuli, s')
        ax2.set_ylabel('Weights, w')
        ax3.set_ylabel('Reward, R')
        ax3.set_xlabel('Trial, t')
        ax2.legend()
        ax3.legend()

        if minW < 0:
            ax2.add_line(plt.Line2D([0, N_trial-1], [0, 0], color='black', linewidth=0.2))
        if minV < 0 or minR < 0:
            ax3.add_line(plt.Line2D([0, N_trial-1], [0, 0], color='black', linewidth=0.2))

        return [ax1, ax2, ax3]

class BaseTDLearner:
    """
    Base class for the Temporal Difference (TD) learning algorithm.
    """
    def __init__(self,gamma=0.5, alpha=0.1, n_states=10,):
        """
        Initialize the TD learning algorithm.
        
        Parameters:
        - gamma: The discount factor.
        - alpha: The learning rate.
        - n_states: The number of states.
        """
        self.V = np.zeros(n_states)
        self.S_prev = None
        self.gamma = gamma
        self.alpha = alpha
        self.n_states = n_states

        #History data (for plotting) 
        self.V_history = []
        self.S_history = []
        self.R_history = []
        self.TD_history = []
        # in it's learning Q values 
        self.Q_history = []
        self.A_history = []

        self.theoretical_value = None # if not None, this is will be plotted ontop 

    def learn_episode(self, 
                    states : np.ndarray, 
                    rewards : np.ndarray,):
            """
            Learn from an episode of states and rewards.
            
            Parameters:
            - states: The states visited in the episode, represented as a numpy array of integers.
            - rewards: The rewards received after each state in the episode, represented as a numpy array of floats.
            """
            states = np.array(states)
            rewards = np.array(rewards)
            assert len(states) == len(rewards), "States and rewards must be the same length"

            T_episode = len(states)+ 1 # get the length of the episode (including the initial None state)

            # Insert an unrewarded "None" state at the beginning to indicate the start of an episode
            rewards = np.insert(rewards, 0, 0)
            states = np.insert(states.astype(object), [0, len(states)], [None, None])

            # Loop over the states and learn from each transition
            TD_errors = np.empty(T_episode) # store the TD errors
            for i in range(T_episode):
                # Learn from this transition
                TD_errors[i] = self.learn(states[i], states[i+1], rewards[i])
            
            # Save to history 
            self.V_history.append(self.V.copy())
            self.S_history.append(states[:T_episode])
            self.TD_history.append(TD_errors)
            self.R_history.append(rewards)
    
    def plot(self, episode=0, axs=None):
        """
        Plot the learning history for a specific episode.
        
        Parameters:
        - episode: The index of the episode to plot.
        - axs: The axes objects to plot on. If None, a new figure and axes will be created.
        
        Returns:
        - The axes objects.
        """
        T = self.S_history[episode].shape[0]
        T_hist = np.array([[S.shape[0] for S in self.S_history]])
        T_plot = int(np.percentile(T_hist, 90))

        # convert state history from integers to one-hot encoding
        S_history = np.array([np.eye(self.n_states)[s] if s is not None else np.zeros(self.n_states) for s in self.S_history[episode]]).T
        max_V, min_V, max_TD, min_TD, max_R, min_R = 0,1,0,1,0,1
        for ep in range(len(self.V_history)):
            max_V = np.ceil(max(max_V, np.max(self.V_history[ep])))
            min_V = np.floor(min(min_V, np.min(self.V_history[ep])))
            max_TD = np.ceil(max(max_TD, np.max(self.TD_history[ep])))
            min_TD = np.floor(min(min_TD, np.min(self.TD_history[ep])))
            max_R = np.ceil(max(max_R, np.max(self.R_history[ep])))
            min_R = np.floor(min(min_R, np.min(self.R_history[ep])))
        
        
        if axs is None:
            fig = plt.figure(figsize=(3, 3))
            gs = gridspec.GridSpec(3, 2, height_ratios=[4, 1, 1], width_ratios=[4, 1], hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
            ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax4 = fig.add_subplot(gs[2, 0], sharex=ax1)
            axs = [ax1, ax2, ax3, ax4]

        [ax1, ax2, ax3, ax4] = axs
        fig = ax1.get_figure()

        for ax in axs: 
            ax.spines['left'].set_position(('outward', 3))
            ax.spines['bottom'].set_position(('outward', 3))

        # Central big plot shows the states 
        ax1.imshow(S_history[:,:T_plot], cmap='Greys', aspect='auto')
        ax1.set_ylabel("State, S", rotation=0, labelpad=20)
        ax1.spines['left'].set_bounds(0, self.n_states-1)
        ax1.spines['bottom'].set_bounds(0, T_plot-1)
        ax1.xaxis.set_visible(False)

        # Right plot sharing y-axis with the central plot
        ax2.fill_betweenx(np.arange(self.n_states), self.V_history[episode], 0, color='C0', alpha=0.5, linewidth=0)
        ax2.spines['left'].set_bounds(0, self.n_states-1)
        ax2.set_xlabel('Value, V(S)')
        ax2.set_xlim([min_V, max_V])
        ax2.yaxis.set_visible(False)
        ax2.set_xticks([0, max_V])
        ax2.set_xticklabels([f'{0.0:.1f}', f'{max_V:.1f}'])
        if min_V < 0: 
            ax2.add_line(plt.Line2D([0, 0], [0, self.n_states-1], color='black', linewidth=0.2))
        if self.theoretical_value is not None:
            ax2.plot(self.theoretical_value, np.arange(self.n_states), color='k', linestyle='--', linewidth=0.5, label='Theory')
            ax2.legend(loc="right")


        # First bottom plot sharing x-axis with the central plot
        # ax3.fill_between(np.arange(T), self.R_history[episode], 0, color='orange', alpha=0.5, linewidth=0)
        ax3.bar(np.arange(T)[:T_plot], self.R_history[episode][:T_plot], color='orange', alpha=0.5, linewidth=0)
        ax3.spines['bottom'].set_bounds(0, T_plot-1)
        ax3.set_ylabel('Reward, R', rotation=0, labelpad=20)
        ax3.set_ylim([min_R, max_R])
        ax3.xaxis.set_visible(False)
        ax3.set_yticks([min_R, 0, max_R])
        if min_R < 0: 
            ax3.add_line(plt.Line2D([0, T_plot-1], [0, 0], color='black', linewidth=0.2))
        
        # Second bottom plot sharing x-axis with the central plot
        x_new = np.linspace(0, T_plot, 1000)
        
        y_new = np.interp(x_new, np.arange(T), self.TD_history[episode], right=0)
        ax4.fill_between(x_new, y_new, 0, where=y_new >= 0, color='green', alpha=0.5, linewidth=0)
        ax4.fill_between(x_new, y_new, 0, where=y_new < 0, color='red', alpha=0.5, linewidth=0)
        ax4.spines['bottom'].set_bounds(0, T_plot-1)
        ax4.set_ylabel('TD error', rotation=0, labelpad=20)
        ax4.set_xlabel('Time step, t')
        ax4.set_ylim([min_TD, max_TD])
        ax4.set_xticks([0, T_plot-1])
        ax4.set_yticks([min_TD, 0, max_TD])
        if min_TD < 0: 
            ax4.add_line(plt.Line2D([0, T_plot-1], [0, 0], color='black', linewidth=0.2))

        fig.suptitle(f'Episode {episode}')
        fig.subplots_adjust(left=0.3, bottom=0.2, right=0.9, top=0.9, wspace=None, hspace=None) #remove border
        return axs
    
    def animate_plot(self):
        """
        Animates the plot of the agent's performance over episodes.

        Returns:
            anim (matplotlib.animation.FuncAnimation): Animation object representing the plot animation.
        """
        n_episodes = len(self.S_history)
        episodes = np.arange(n_episodes)

        def update(episode, axs):
            for ax in axs:
                ax.clear()
            axs = self.plot(episode=episode, axs=axs)
            return axs

        axs = self.plot(episode=0, axs=None)
        anim = FuncAnimation(plt.gcf(), update, frames=episodes, fargs=(axs,), interval=100)
        plt.close()
        return anim
        
class BaseTDQLearner(BaseTDLearner):
    """
    Base class for Temporal Difference Q-Learning algorithms.

    Args:
        gamma (float): Discount factor for future rewards. Default is 0.5.
        alpha (float): Learning rate. Default is 0.1.
        n_states (int): Number of states in the environment. Default is 10.
        n_actions (int): Number of actions in the environment. Default is 4.
    """

    def __init__(self, gamma=0.5, alpha=0.1, n_states=10, n_actions=4):
        super(BaseTDQLearner, self).__init__(gamma=gamma, alpha=alpha, n_states=n_states)

    def learn_episode(
        self,
        states : np.ndarray, 
        actions : np.ndarray,
        rewards : np.ndarray,
    ):
        """
        Learn from a single episode of transitions.

        Args:
            states (np.ndarray): Array of states visited during the episode.
            actions (np.ndarray): Array of actions taken during the episode.
            rewards (np.ndarray): Array of rewards received during the episode.
        """

        T_episode = len(states) + 1 # get the length of the episode (including the initial None state)

        # Insert an unrewarded "None" state at the beginning to indicate the start of an episode
        rewards = np.insert(rewards, 0, 0)
        states = np.insert(states.astype(object), [0, len(states)], [None, None])
        actions = np.insert(actions.astype(object), [0, len(actions)], [None, None])

        # Loop over the states and learn from each transition
        TD_errors = np.empty(T_episode) # store the TD errors
        for i in range(T_episode):
            # Learn from this transition
            TD_errors[i] = self.learn(states[i], states[i+1], actions[i], actions[i+1], rewards[i])
        
        # Save to history 
        self.Q_history.append(self.Q.copy())
        self.S_history.append(states[:T_episode])
        self.A_history.append(states[:T_episode])
        self.TD_history.append(TD_errors)
        self.R_history.append(rewards)
        self.V_history.append(np.mean(self.Q, axis=1))




class MiniGrid():
    """
    Represents a grid environment for reinforcement learning.

    Parameters:
    - grid : np.ndarray, optional
        The grid representation of the environment.
    - reward_locations : list, optional
        The locations of the rewards in the grid. Default is [(5,5)].
    - reward_values : list, optional
        The values of the rewards. Default is [10].
    - max_steps : int, optional
        The maximum number of steps allowed in an episode. Default is 100.
    - action_dict : dict, optional
        A dictionary mapping action indices to action labels. Default is small_action_dict.
    - cost_per_step : float, optional
        The cost incurred per step. Default is 0.1.
    - cost_per_wall_collision : float, optional
        The cost incurred per wall collision. Default is 1.
    """

    def __init__(
        self,
        grid : np.ndarray = None,
        reward_locations : list = [(5,5)],
        reward_values : list = [10],
        max_steps = 100,
        action_dict = small_action_dict, 
        cost_per_step = 0.1,
        cost_per_wall_collision = 1,
    ):
        """
        Initializes the MiniGrid environment.

        Parameters:
        - grid : np.ndarray, optional
            The grid representation of the environment. 
        - reward_locations : list, optional
            The locations of the rewards in the grid. Default is [(5,5)].
        - reward_values : list, optional
            The values of the rewards. Default is [10].
        - max_steps : int, optional
            The maximum number of steps allowed in an episode. Default is 100.
        - action_dict : dict, optional
            A dictionary mapping action indices to action labels. Default is small_action_dict.
        - cost_per_step : float, optional
            The cost incurred per step. Default is 0.1.
        - cost_per_wall_collision : float, optional
            The cost incurred per wall collision. Default is 1.
        """
        
        if grid is not None:
            self.grid = grid
            self.width = grid.shape[1]
            self.height = grid.shape[0]
            self.x_coords = np.arange(self.width)
            self.y_coords = np.arange(self.height)[::-1]
            self.grid_shape = (self.height, self.width)
            self.n_states = self.width * self.height
        
        self.reward_locations = reward_locations
        self.reward_values = reward_values
        assert len(reward_locations) == len(reward_values), "Reward locations and values must be the same length"

        self.cost_per_step = cost_per_step
        self.cost_per_wall_collision = cost_per_wall_collision
        self.max_steps = max_steps
        self.reward_extent = 1 #effective size of reward is 1 (1 grid)
        self.agent_extent = 1 #effective size of agent is 1 (1 grid)
        self.episode_number = 0 
        self.recent_episode_length = self.max_steps
        self.episode_history = {}
        self.av_time_per_episode = self.max_steps
        self.episode_lengths = []

        self.action_dict = action_dict
        self.n_actions = len(self.action_dict)

        self.reset()

    def step(self, action):
        """
        Takes a step in the environment based on the given action.

        Parameters:
        - action : int
            The index of the action to take.

        Raises:
        - NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def reset(self):
        """
        Resets the environment by placing the agent at a random position and direction.
        """
        is_illegal_position = True
        while is_illegal_position:
            self.agent_pos = (np.random.randint(self.width), np.random.randint(self.height))
            is_illegal_position = self.is_wall(self.agent_pos)
        self.agent_direction = np.random.randint(self.n_actions)

    def is_wall(self, pos : tuple): 
        """
        Checks if the given position is a wall.

        Parameters:
        - pos : tuple
            The position to check.

        Returns:
        - bool: True if the position is a wall, False otherwise.
        """
        (x, y) = pos
        return self.grid[self.height - 1 - y, x] # y is flipped because of the way the grid is plotted
        
    def get_reward(self, pos : tuple):
        """
        Gets the reward value at the given position.

        Parameters:
        - pos : tuple
            The position to get the reward value for.

        Returns:
        - float: The reward value at the given position.
        """
        if pos in self.reward_locations:
            return self.reward_values[self.reward_locations.index(pos)]
        else:
            return 0
    
    def plot_episode(self, episode_number=None, ax=None, stop_at_step=None):
        """
        Plots the trajectory of an episode.

        Parameters:
        - episode_number : int, optional
            The episode number to plot. If not provided, the most recent episode will be plotted.
        - ax : matplotlib.axes.Axes, optional
            The axes to plot on. If not provided, a new figure will be created.
        - stop_at_step : int, optional
            The step at which to stop plotting. If not provided, the entire episode will be plotted.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        if episode_number is None:
            episode_number = self.episode_number - 1
        else: 
            episode_number = np.arange(self.episode_number)[episode_number]
        episode_data = self.episode_history[episode_number]
        positions = episode_data["positions"]
        actions = episode_data["actions"]
        if ax is None:
            ax = self._plot_env(ax=ax)
            ax = self._plot_rewards(ax)
        if stop_at_step is None: stop_at_step = len(positions)
        for i in range(stop_at_step):
            episode_frac = i/len(positions)
            ax = self._plot_agent(agent_pos=positions[i], agent_direction=actions[i], ax=ax, color = plt.cm.viridis(1-episode_frac), size_scaler=(1+episode_frac)/2, alpha = (1+episode_frac)/2)
            if i > 0:
                ax.plot([positions[i-1][0], positions[i][0]], [positions[i-1][1], positions[i][1]], color=plt.cm.viridis(1-episode_frac), alpha = (1+episode_frac)/2, linewidth=0.5)
        
        return ax

    def plot_Q(self, Q_values):
        """
        Plots the Q-values of the environment.

        Parameters:
        - Q_values : np.ndarray
            The Q-values of the environment.

        Returns:
        - list: The list of axes objects containing the plots.
        """
        assert (Q_values.shape == (self.n_states, self.n_actions)), f"Q_values must be of shape (n_states, n_actions) = ({self.n_states}, {self.n_actions}) but got {Q_values.shape}"
        
        # Q_values is shape (n_states, n_actions) so we need to reshape it to (height, width, n_actions). However by default numpy.reshape reads the array in row-major order, so we need to transpose the first two dimensions

        Q_values_grid = Q_values.reshape(self.grid_shape[0], self.grid_shape[1], self.n_actions)

        min_Q = np.min(Q_values)
        max_Q = np.max(Q_values)

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(3, 3, hspace=0.15, wspace=0.15)
        axN = fig.add_subplot(gs[0, 1])
        axW = fig.add_subplot(gs[1, 0])
        axE = fig.add_subplot(gs[1, 2])
        axS = fig.add_subplot(gs[2, 1])
        axC = fig.add_subplot(gs[1, 1])
        axs = [axN, axE, axS, axW, axC]
        for i in range(self.n_actions):
            axs[i] = self._plot_env(ax=axs[i])
            axs[i] = self._plot_rewards(ax=axs[i])
            axs[i].imshow(Q_values_grid[:,:,i], cmap="viridis", extent=[-0.5, self.width-0.5, -0.5, self.height-0.5], alpha=0.5, vmax=max_Q, vmin=min_Q)
            axs[i].set_title(f"Q(s,{self.action_dict[i]['label']})", color=plt.colormaps['twilight'](i / self.n_actions))
            if i > 0:
                axs[i].set_yticks([])
                axs[i].set_ylabel("")
        axC = self._plot_env(ax=axC)
        axC = self._plot_rewards(ax=axC)
        axC = self.plot_policy(Q_values, ax=axC)
        axC.set_title("π(s)")
        return axs
    
    def plot_training(self, ax=None):
        """
        Plots the training progress.

        Parameters:
        - ax : matplotlib.axes.Axes, optional
            The axes to plot on. If not provided, a new figure will be created.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 1))
        episode_lengths = np.array([len(self.episode_history[i]["states"]) for i in range(self.episode_number)])
        n_eps = len(episode_lengths)
        smoothed_episode_lengths = [np.mean(episode_lengths[max(0, i-100):i+1]) for i in range(n_eps)]
        ax.scatter(np.arange(self.episode_number), episode_lengths, linewidth=0, alpha=0.5,c='C0', label="Episode length")
        ax.plot(np.arange(len(smoothed_episode_lengths)), smoothed_episode_lengths, color='k',linestyle="--",linewidth=0.5, label = "Smoothed")
        ax.set_xlabel("Episode")
        ax.legend()
        ax = format_axes(ax)

        return ax
    
    def render(self, ax=None):
        """
        Renders the current state of the environment.

        Parameters:
        - ax : matplotlib.axes.Axes, optional
            The axes to plot on. If not provided, a new figure will be created.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        ax = self._plot_env(ax=ax)
        ax = self._plot_agent(ax=ax,agent_pos=self.agent_pos, agent_direction=self.agent_direction)
        ax = self._plot_rewards(ax)
        return ax 

    def plot_first_and_last_5_episodes(self):
        """
        Plots the trajectories of the first and last 5 episodes.

        Returns:
        - numpy.ndarray: The array of axes objects containing the plots.
        """
        fig, ax = plt.subplots(2,5, figsize=(10,4))
        for i in range(5):

            ax[0,i] = self._plot_env(ax=ax[0,i])
            ax[1,i] = self._plot_env(ax=ax[1,i])
            ax[0,i] = self._plot_rewards(ax=ax[0,i])
            ax[1,i] = self._plot_rewards(ax=ax[1,i])
        
            self.plot_episode(i, ax=ax[0,i])
            self.plot_episode(-i-1, ax=ax[1,i])
            if i == 2: 
                ax[0,i].set_title("First 5 episodes")
                ax[1,i].set_title("Last 5 episodes")
        return ax
    
    def _plot_env(self, ax=None):
        """
        Plots the environment grid.

        Parameters:
        - ax : matplotlib.axes.Axes, optional
            The axes to plot on. If not provided, a new figure will be created.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.1*self.width, 0.1*self.height))
        ax.imshow(self.grid, cmap="Greys", zorder=20, alpha=self.grid.astype(float), extent=[-0.5, self.width-0.5, -0.5, self.height-0.5])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        return ax 

    def _plot_rewards(self, ax, **kwargs):
        """
        Plots the rewards in the environment.

        Parameters:
        - ax : matplotlib.axes.Axes
            The axes to plot on.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        for (i,reward) in enumerate(self.reward_locations):
            reward_value = self.reward_values[i]
            max_rv, min_rv = np.max(self.reward_values), np.min(self.reward_values)
            # color = plt.colormaps['plasma']((reward_value - min_rv) / max(1, max_rv - min_rv))
            reward_patch = plt.Circle(reward, self.reward_extent * (0.4 + 0.3*((reward_value - min_rv) / max(1, max_rv - min_rv))), color='orange',zorder=2)
            ax.add_patch(reward_patch)
            ax.text(reward[0], reward[1], "R", color='white', weight="bold",fontsize=4, ha='center', va='center',alpha=0.5)
        return ax
    
    def _plot_arrow(self, pos, direction, ax, **kwargs):
        """
        Plots an arrow indicating the agent's direction.

        Parameters:
        - pos : tuple
            The position of the arrow.
        - direction : int
            The direction of the arrow.
        - ax : matplotlib.axes.Axes
            The axes to plot on.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        alpha = kwargs.get("alpha", 1)
        color = kwargs.get("color", plt.cm.viridis(0.5))
        size_scaler = kwargs.get("size_scaler", 1)
        # Define the base triangle vertices relative to origin
        base_triangle = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]) - np.array([0.5,0.5])
        scaled_triangle = base_triangle * size_scaler
        # Define the rotation matrix based on direction
        angle = (2*np.pi - (direction/self.n_actions)*2*np.pi) # assumes action evenly spaced around circle
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                    [np.sin(angle), np.cos(angle)]])
        # Rotate the triangle
        rotated_triangle = np.dot(scaled_triangle, rotation_matrix.T)   
        # Translate the triangle to the agent position
        translated_triangle = rotated_triangle + np.array(pos)
        # Plot the triangle
        triangle = plt.Polygon(translated_triangle, color=color, alpha=alpha, linewidth=0)
        ax.add_patch(triangle)
        return ax

    def _plot_circle(self, pos, ax, **kwargs):
        """
        Plots a circle representing a position in the environment.

        Parameters:
        - pos : tuple
            The position of the circle.
        - ax : matplotlib.axes.Axes
            The axes to plot on.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        alpha = kwargs.get("alpha", 1)
        color = kwargs.get("color", plt.cm.viridis(0.5))
        size_scaler = 0.5*kwargs.get("size_scaler", 1)
        circle = plt.Circle(pos, size_scaler, color=color, alpha=alpha, linewidth=0)
        ax.add_patch(circle)
        return ax
    
    def _plot_agent(self, ax, agent_pos, agent_direction,  **kwargs):
        """
        Plots the agent in the environment.

        Parameters:
        - ax : matplotlib.axes.Axes
            The axes to plot on.
        - agent_pos : tuple
            The position of the agent.
        - agent_direction : int
            The direction of the agent.

        Returns:
        - matplotlib.axes.Axes: The axes object containing the plot.
        """
        size_scaler = kwargs.pop("size_scaler", 1)
        size_scaler *= self.agent_extent
        ax = self._plot_circle(pos=agent_pos, direction=agent_direction, ax=ax, size_scaler=size_scaler, **kwargs)
        return ax


    def animate_episodes(self,episodes=None, **kwargs):
        if episodes is None:
            episode = [self.episode_number - 1]
        def update(frames, ax):
            episode, frame = frames
            ax.clear()
            ax = self._plot_env(ax=ax)
            ax = self._plot_rewards(ax)
            ax = self.plot_policy(**kwargs, ax=ax)
            ax = self.plot_episode(episode_number=episode, ax=ax, stop_at_step=frame)
            plt.close()
            return ax
        
        axs = self.plot_Q(**kwargs)
        ep_fr = [(episode, frame) for episode in episodes for frame in range(len(self.
        episode_history[episode]["positions"]))]

        anim = FuncAnimation(plt.gcf(), update, frames=ep_fr, fargs=(axs[-1],), interval=100)        
        plt.close()
        return anim
        
    
    def plot_policy(self, Q_values, ax=None): 
        Q_values = Q_values.reshape(self.grid_shape[0], self.grid_shape[1], self.n_actions)
        optimal_actions = Q_values.argmax(axis=2)
        if ax is None:
            ax = self._plot_env()
            ax = self._plot_rewards(ax)
        for x in self.x_coords:
            for y in self.y_coords:
                if self.is_wall((x,y)):
                    continue
                action = optimal_actions[self.height-1-y,x]
                color = plt.colormaps['twilight'](action / self.n_actions)
                ax = self._plot_arrow(pos=(x,y), direction=action, ax=ax, color = color, size_scaler=0.5*self.agent_extent)
        
        return ax

    def pos_to_state(self, pos):
        return (self.height - 1 - pos[1]) * self.width + pos[0] 
    






class MiniSpace(MiniGrid):
    def __init__(
        self,
        env : ratinabox.Environment,
        ag : ratinabox.Agent,
        state_features : ratinabox.Neurons,
        reward_locations : list = [(0.5,0.5)],
        reward_values : list = [10,],
        max_steps : int = 100,
        action_dict = large_action_dict,
    ):
        # Initialize the ratinabox environment
        self.env = env
        self.ag = ag
        self.state_features = state_features

        MiniGrid.__init__(self,
                          grid=np.array([[0]]), 
                          reward_locations=reward_locations, 
                          reward_values=reward_values, 
                          max_steps=max_steps,
                          action_dict=action_dict)
         
        self.reward_extent = 0.1
        self.agent_extent = 0.05
        self.delta_x = 0.05
        return 
    

    def pos_to_state(self, pos):
        pos = np.array(pos)
        state = self.state_features.get_state(evaluate_at=None, pos=pos)
        if pos.ndim == 1:
            return state[:,0] # (n_features,)
        else:
            return state.T # (n_positions, n_features,)
    
    def step(self, action):
        # Propose a new position based on the action
        proposed_new_pos = None # write this 
        delta = self.action_dict[action]['delta']
    
        # Get the proposed next position by adding the delta to the current position
        proposed_new_pos = np.array(self.agent_pos) + self.delta_x*np.array(delta)
        proposed_new_pos = tuple(proposed_new_pos)
        # Check if the new position is a wall or reward 
        step = np.array([np.array(self.agent_pos),np.array(proposed_new_pos)])
        is_wall = (True in self.env.check_wall_collisions(step)[1]) # returns True if the proposed new position crosses a wall
        new_pos = self.agent_pos
        if not is_wall: new_pos = proposed_new_pos
        self.agent_pos = new_pos
        self.agent_direction = action
        # self.ag.update(forced_next_step=np.array(new_pos)) # internal RiaB func. to move the agent
        # Check if the new position is a reward
        reward = self.get_reward(new_pos) # returns True if the proposed new position is a reward
        is_terminal = (reward > 0) # If a reward is found then the episode is over
        reward += -1 # cost of moving 
        if is_wall: reward += -1

        # Get the new state 
        # self.state_features.update() # internal RiaB func. to update the state features
        # state = self.state_features.firingrate.copy()
        state = self.pos_to_state(new_pos)

        return state, reward, is_terminal
    
    def reset(self):
        self.agent_pos = tuple(self.env.sample_positions(1)[0])
    
    def get_reward(self, pos):
        for (reward, reward_value) in zip(self.reward_locations, self.reward_values):
            if np.linalg.norm(np.array(reward) - np.array(pos)) < self.reward_extent:
                return reward_value
        return 0


    def _plot_env(self, ax=None):
        if ax is None: 
            fig, ax = self.env.plot_environment(autosave=False)
        else: 
            fig, ax = self.env.plot_environment(ax=ax, fig = ax.get_figure())
        return ax
    
    def plot_state_features(self,**kwargs):
        fig, ax = self.state_features.plot_rate_map(**kwargs)
        return ax 
    
    def plot_policy(self, Q_function, ax=None):
        positions = self.env.discretise_environment(dx=0.05).reshape(-1,2) 
        states = self.pos_to_state(positions)
        Q_values = Q_function(states)
        optimal_actions = np.argmax(Q_values, axis=1)
        if ax is None:
            ax = self._plot_env()
            ax = self._plot_rewards(ax)
        for (i,Qs) in enumerate(Q_values):
            pos = tuple(positions[i])
            action = optimal_actions[i]
            color = plt.colormaps['twilight'](action / self.n_actions)
            ax = self._plot_arrow(pos=pos, direction=action, ax=ax, color = color, size_scaler=self.agent_extent*0.5)
        ax.set_title(r"$\pi(s)$")

        return ax
    
    def plot_Q(self, Q_function):
        positions = self.env.discretise_environment(dx=0.015)
        positions_shape = positions.shape[:-1]
        states = self.pos_to_state(positions).reshape(*positions_shape,-1)
        Q_values = Q_function(states)

        min_Q = np.min(Q_values)
        max_Q = np.max(Q_values)

        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(3, 3, hspace=0.15, wspace=0.15, left=0.1, right=0.9, top=0.9, bottom=0.1)
        axN = fig.add_subplot(gs[0, 1])
        axNE = fig.add_subplot(gs[0, 2])
        axE = fig.add_subplot(gs[1, 2])
        axSE = fig.add_subplot(gs[2, 2])
        axS = fig.add_subplot(gs[2, 1])
        axSW = fig.add_subplot(gs[2, 0])
        axW = fig.add_subplot(gs[1, 0])
        axNW = fig.add_subplot(gs[0, 0])
        axC = fig.add_subplot(gs[1, 1])
        axs = [axN, axNE, axE, axSE, axS, axSW, axW, axNW, axC]
        for i in range(self.n_actions):
            axs[i] = self._plot_env(ax=axs[i])
            axs[i] = self._plot_rewards(ax=axs[i])
            axs[i].imshow(Q_values[:,:,i], cmap="viridis", extent=self.env.extent, alpha=0.5, vmax=max_Q, vmin=min_Q)
            axs[i].set_title(rf"$Q(s,{self.action_dict[i]['label']})$", color=plt.colormaps['twilight'](i / self.n_actions))
            if i > 0:
                axs[i].set_yticks([])
                axs[i].set_ylabel("")
        axC = self._plot_env(ax=axC)
        axC = self._plot_rewards(ax=axC)
        # axC = self._plot_optimal_Q(Q_function, ax=axC)
        axC = self.plot_policy(Q_function, ax=axC)

        return axs
    
    def _plot_optimal_Q(self, Q_function, ax=None):
        positions = self.env.discretise_environment(dx=0.025)
        positions_shape = positions.shape[:-1]
        states = self.pos_to_state(positions).reshape(*positions_shape,-1)
        Q_values = Q_function(states)
        min_Q = np.min(Q_values)
        max_Q = np.max(Q_values)
        optimal_Q = np.max(Q_values, axis=2)

        optimal_actions = np.argmax(Q_values, axis=2)
        if ax is None:
            ax = self._plot_env()
            ax = self._plot_rewards(ax)
        ax.imshow(optimal_Q, cmap="viridis", extent=self.env.extent, alpha=0.5, vmax=max_Q, vmin=min_Q)
        return ax

    
    def train(
        self,
        tdqlearner, 
        n_episodes=1000,
        max_episode_length=100,
        policy=None,
        ): 

        for i in (pbar := tqdm(range(n_episodes))):

            try: # this just allows you to stop the loop by pressing the stop button in the notebook
                
                # Initialise an episode: 
                terminal = False
                self.reset() # reset the environment
                state = self.pos_to_state(self.agent_pos)
                action = policy(tdqlearner.Q(state, action=None))

                episode_data = {'positions': [], 'states':[], 'actions':[], 'rewards':[]}
                
                step = 0
                while not terminal and step < max_episode_length:
                    # Get the next state and reward using the step() method
                    state_next, reward, terminal = self.step(action)
                    
                    # Get the next action using the policy() function
                    next_action = policy(tdqlearner.Q(state_next, action=None))
                    
                    # Learn from the transition using the tdqlearner.learn() method
                    tdqlearner.learn(state, state_next, action, next_action, reward)

                    # store the data
                    episode_data['positions'].append(self.agent_pos)
                    episode_data['states'].append(state)
                    episode_data['actions'].append(action)
                    episode_data['rewards'].append(reward)

                    # update the state and action
                    state = state_next
                    action = next_action 
                    step += 1

                
                ep_length = step
                self.av_time_per_episode = 0.99*self.av_time_per_episode + 0.01*ep_length
                pbar.set_description(f"Episode {i+1}, Episode length (recent average) {self.av_time_per_episode:.1f}")                
                self.episode_history[self.episode_number] = episode_data
                self.episode_number += 1

            except KeyboardInterrupt:
                break


# import torch 
# # make this an exercise 
# class DNNTDQLearner(torch.nn.Module):
#     def __init__(self, gamma=0.5, alpha=0.1, n_features=10, n_actions=8, hidden_units=(200,200,)):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.n_features = n_features

#         # Initialize the weights
#         self.linear1 = torch.nn.Linear(n_features, hidden_units[0])
#         self.linear2 = torch.nn.Linear(hidden_units[0], hidden_units[1])
#         self.linear3 = torch.nn.Linear(hidden_units[1], n_actions)
#         self.relu = torch.nn.ReLU()

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

#     def forward(self, state):
#         x = self.linear1(state)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.relu(x)
#         x = self.linear3(x)
#         return x
    
#     def Q(self, state, action=None, return_numpy=True):
#         """
#         This function should return the Q value for a given state and action
#         State should be a vector of features. Optionally it can be a batch of states where the batch dimension is the first dimension.
#         If action is None then the function should return the Q values for all actions in the state.
#         """
#         state = torch.tensor(state, dtype=torch.float32)
#         Q_values = self.forward(state)
#         if action is not None:
#             Q_values = Q_values[...,action]
#         if return_numpy:
#             return Q_values.detach().numpy()
#         return Q_values
    
#     def learn(self, S, S_next, A, A_next, R):
#         # Get's the value of the current and next state
#         Q = self.Q(S,A,return_numpy=False) if S is not None else 0
#         Q_next = self.Q(S_next, A_next,return_numpy=False) if S_next is not None else 0
#         # Calculate the gradients using backprop
#         self.zero_grad()
#         Q_target = R + self.gamma * Q_next
#         Q_target = Q_target.detach()
#         loss = torch.nn.functional.mse_loss(Q, Q_target)
#         loss.backward()

#         # Update the weights

#         self.optimizer.step()