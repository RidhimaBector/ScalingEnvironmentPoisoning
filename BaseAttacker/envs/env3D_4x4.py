# Attack action space U: Box=(-10, +10, (16, ), float 32) -- continuous

# import io
# import sys
import math
import os
import sys
from typing import Tuple

import numpy as np
from gym import spaces
from gym.utils import seeding

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *
from envs.victim_environment import VictimEnvironment

# STILL = 0
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

GRID_DIMENSIONS = (4,4) #(4, 4)

"""def categorical_sample(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n>np_random.rand()).argmax()"""


class Grid3D(VictimEnvironment):
    """
    GridWorld_3D_env is a class that implements the GridWorld_3D environment.
    """

    """def _generate_altitude(self, shape):

        s = 1
        wsize = shape[0]
        A = wsize*np.random.randn(s,s)

        # generate land
        while s < wsize:
            s = s*2
            B = (wsize/s)*np.random.rand(s,s)
            for x in range(0, s):
                for y in range(0, s):
                    B[x][y] = B[x][y] + A[math.floor(x/2)][math.floor(y/2)]
            A = B

        # diffuse
        B = A
        for d in range(0,2):
            for x in range(0, wsize):
                for y in range(0, wsize):
                    if (x==0 or x==(wsize-1)) and (y==0 or y==(wsize-1)):
                        pass
                    elif x==0 or x==wsize-1:
                        B[x][y] = np.mean(A[x, y-1:y+2])
                    elif y==0 or y==wsize-1:
                        B[x][y] = np.mean(A[x-1:x+2, y])
                    else:
                        B[x][y] = np.mean(A[x-1:x+2, y-1:y+2])
            A = B

        # make 10 different altitute levels
        A = A - np.min(A)
        A = 10*A/np.max(A)
        A = np.round(A)

        return A"""


    def _generate_altitude(self) -> np.ndarray:
        rows, cols = self.shape
        A = np.zeros(self.shape)

        # Create a gradient from top-left to bottom-right
        # Using linear space to generate values between 8 and 2
        row_values = np.linspace(8, 2, rows)

        # Fill the array with a gradient pattern
        for i in range(rows):
            # Create a row that gradually increases from left to right
            # but maintains the general descent from top to bottom
            base_value = row_values[i]
            # Add some variation within each row while keeping values mostly in 4-8 range
            row = np.linspace(base_value - 2, base_value + 2, cols)
            # Ensure the rightmost column maintains higher elevation
            row[-1] = 8.0  # Keep the right edge at elevation 8
            A[i] = row

        # Clip values to ensure they stay within 0-10 range
        A = np.clip(A, 0, 10)
        # Round to nearest integer
        A = np.round(A).astype(int)

        return A


    def _base_altitude(self) -> np.ndarray:
        A = np.zeros(self.shape)
        #A[0] = [8, 7, 5, 4]
        #A[1] = [9, 6, 4, 2]
        #A[2] = [8, 5, 4, 1]
        #A[3] = [8, 4, 2, 0]
        A[0] = [8.00, 8.00, 8.00, 8.00] #[8, 7, 5, 4]
        A[1] = [6.00, 6.00, 6.00, 8.00] #[9, 6, 4, 2]
        A[2] = [4.00, 4.00, 6.00, 8.00] #[8, 5, 4, 1]
        A[3] = [2.00, 4.00, 6.00, 8.00] #[8, 4, 2, 0]

        """A[0] = [8.00, 8.25, 8.50, 8.75] #[8, 7, 5, 4]
        A[1] = [6.00, 6.00, 6.00, 9.00] #[9, 6, 4, 2]
        A[2] = [5.00, 5.00, 5.00, 9.25] #[8, 5, 4, 1]
        A[3] = [4.00, 3.00, 2.00, 1.00] """#[8, 4, 2, 0]

        return A


    def _defined_altitude(self) -> np.ndarray:
        if GRID_DIMENSIONS == (4,4):
            return self._base_altitude()
        else:
            return self._generate_altitude()


    def _calculate_dynamics(self, shape, nA, nS, A):
        # Base probability of struggling uphill
        Bstrug = 0.9
        # Base probability of stilding downhill
        Bslide = 0.2

        wsize = shape[0]

        T = np.zeros((nA, nS, nS))
        for a in range(nA):
            T[a] = np.eye(nS)

        for row in range(0, wsize):
            for col in range(0, wsize):

                # -- current state --
                s1 = row*wsize + col

                # -- check north --
                if row > 0:
                    s2 = (row-1)*wsize + col
                    s3 = (row-2)*wsize + col

                    # slope
                    diff = A[row-1][col] - A[row][col]

                    if row > 1:
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0
                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[0][s1][s1] = stay
                    T[0][s1][s2] = move
                    if row > 1:
                        T[0][s1][s3] = slip

                # -- check east --
                if col < wsize-1 :
                    s2 = row*wsize + col + 1
                    s3 = row*wsize + col + 2

                    # slope
                    diff = A[row][col+1] - A[row][col]

                    if col < wsize-2 :
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0
                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[1][s1][s1] = stay
                    T[1][s1][s2] = move
                    if col < wsize-2:
                        T[1][s1][s3] = slip

                # -- check south --
                if row < wsize-1:
                    s2 = (row+1)*wsize + col
                    s3 = (row+2)*wsize + col

                    # slope
                    diff = A[row+1][col] - A[row][col]

                    if (row+2) < wsize:
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0

                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[2][s1][s1] = stay
                    T[2][s1][s2] = move
                    if row+2 < wsize:
                        T[2][s1][s3] = slip

                # -- check west --
                if col>0:
                    s2 = row*wsize + col - 1
                    s3 = row*wsize + col - 2

                    # slope
                    diff = A[row][col-1] - A[row][col]

                    if col > 1:
                        slip = Bslide/(1+math.exp(2+2*diff))
                    else:
                        slip = 0

                    stay = Bstrug/(1+math.exp(2-2*diff))
                    move = 1 - slip - stay

                    T[3][s1][s1] = stay
                    T[3][s1][s2] = move
                    if col > 1:
                        T[3][s1][s3] = slip

        small = 0.000001
        for a in range(nA):
            T[a] = (T[a]+small)/(1+nS*small)

        return T


    def _calculate_transition_status(self, cur_s, action, T):
        prob = T[action][cur_s]
        next_s = np.random.choice(np.arange(len(prob)), p=prob)
        prob_next_s = prob[next_s]

        target_state = np.ravel_multi_index((self.shape[0]-1, self.shape[1]-1), self.shape)
        is_done = next_s == target_state
        return prob_next_s, next_s, -1.0, is_done


    """def _calculate_P(self, a, nS, T):
        P = {}
        for s in range(nS):
            P[s] = {a : [] for a in range(nA)}
            P[s][NORTH] = self._calculate_transition_status(s, NORTH, T)
            P[s][EAST] = self._calculate_transition_status(s, EAST, T)
            P[s][SOUTH] = self._calculate_transition_status(s, SOUTH, T)
            P[s][WEST] = self._calculate_transition_status(s, WEST, T)

        return P"""


    def __init__(self, grid_dimensions: Tuple[int, int] = GRID_DIMENSIONS):
        super().__init__()
        self.shape = grid_dimensions

        self.nS = np.prod(self.shape)
        self.nA = 4

        # generate altitute
        self.altitude_default = self._defined_altitude()

        self._altitude = self.altitude_default.copy()

        # environment transition
        self.T = self._calculate_dynamics(self.shape, self.nA, self.nS, self.altitude)
        self.INIT_T = self.T.copy()

        # always start in state (0,0)
        #self.isd = np.zeros(self.nS)
        #self.isd[0] = 1.0 #[np.ravel_multi_index((0,0), self.shape)] = 1.0
        # target state
        #self.goal_s = 12 # Victim Goal State               #self.target_s = 12 #np.ravel_multi_index((3,3), self.shape) #Change_Target_Behavior #E-(3,2), M-(3,3), Mp-(3,0), H-(3,0)

        self.lastaction = None # for rendering
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = 0 #categorical_sample(self.isd, self.np_random)


    def render_altitude_heatmap(self) -> np.ndarray:
        """Returns a numpy array representing the environment as an RGB heatmap based on altitude.
        Higher altitudes are represented by hotter colors (red), lower altitudes by cooler colors (blue).

        Returns:
            np.ndarray: RGB array of shape (height, width, 3) with values in range [0,1]
        """
        # Initialize RGB array
        rgb_map = np.zeros((*self.shape, 3))

        # Normalize altitude to 0-1 range
        norm_altitude = self.altitude / 10.0  # Since altitude is 0-10

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Create smooth transition from blue (cold) to red (hot)
                # Blue decreases as altitude increases
                rgb_map[i,j,2] = max(0, 1 - 2 * norm_altitude[i,j])
                # Red increases as altitude increases
                rgb_map[i,j,0] = max(0, 2 * norm_altitude[i,j] - 1)
                # Green creates yellow transition in middle ranges
                rgb_map[i,j,1] = max(0, 1 - abs(2 * norm_altitude[i,j] - 1))

        return rgb_map


    def render_grid(self, rgb_map: np.ndarray) -> np.ndarray:
        """Creates a visual grid representation from the RGB heatmap with cell borders.

        Args:
            rgb_map: RGB array of shape (height, width, 3) with values in range [0,1]

        Returns:
            np.ndarray: RGB array of shape (height*cell_size, width*cell_size, 3) representing
                       the grid with borders and colored cells
        """
        cell_size = 50  # Size of each cell in pixels
        border_width = 2  # Width of cell borders in pixels

        # Create output image array
        height, width = self.shape
        img = np.zeros((height * cell_size, width * cell_size, 3))

        # Fill cells with colors from heatmap
        for i in range(height):
            for j in range(width):
                # Calculate pixel coordinates
                top = i * cell_size
                left = j * cell_size

                # Fill cell interior with heatmap color
                img[top:top+cell_size, left:left+cell_size] = rgb_map[i,j]

                # Add black borders
                img[top:top+border_width, left:left+cell_size] = 0  # Top border
                img[top+cell_size-border_width:top+cell_size, left:left+cell_size] = 0  # Bottom border
                img[top:top+cell_size, left:left+border_width] = 0  # Left border
                img[top:top+cell_size, left+cell_size-border_width:left+cell_size] = 0  # Right border

                # Mark current state position with white dot if this is current state
                if self.s == i * width + j:
                    center_y = top + cell_size // 2
                    center_x = left + cell_size // 2
                    dot_radius = 5
                    y_indices, x_indices = np.ogrid[center_y-dot_radius:center_y+dot_radius,
                                                  center_x-dot_radius:center_x+dot_radius]
                    circle_mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= dot_radius**2
                    img[y_indices.min():y_indices.max()+1,
                        x_indices.min():x_indices.max()+1][circle_mask] = 1.0

        return img

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        self._altitude = value

    @property
    def env_dynamics(self):
        return self.altitude.copy().reshape((self.nS, 1))


    @property
    def max_steps(self):
        return None

    def reset(self):
        self.s = 0 #categorical_sample(self.isd, self.np_random) #start state
        self.lastaction = None
        return self.s


    def reset_altitude(self):
        self.altitude = self.altitude_default.copy()
        self.T = self._calculate_dynamics(self.shape, self.nA, self.nS, self.altitude)
        #return self.altitude


    def step(self, action):
        prob, next_state, reward, done = self._calculate_transition_status(self.s, action, self.T) #P[self.s][action]
        #i = categorical_sample([t[0] for t in transitions], self.np_random)
        #p, s, r, d = transition[0]
        self.s = next_state
        self.lastaction = action
        return (next_state, reward, done, {"prob" : prob})


    """def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == (3,3): #Change_Target_Behavior #E-(3,2), M-(3,3), Mp-(3,0), H-(3,0)
                output = " T "
            else:
                output = " o "

            if position[1]==0:
                output = output.lstrip()
            if position[1]==self.shape[1]-1:
                output = output.rstrip()
                output += "\n"

            #outfile.write(output)

        #outfile.write("\n")"""


if __name__ == "__main__":
    env = Grid3D(grid_dimensions=(5,5))
    altitude = env.altitude
    print("Altitude matrix:")
    print(altitude)

    for col, row in enumerate(altitude):
        print(f"Column {col}: {row}")

    altitude_heatmap = env.render_altitude_heatmap()
    for i in range(5):
        print()
        for j in range(5):
            print("(", end="")
            for k in range(3):
                print(altitude_heatmap[i][j][k], end=" ")
            print(")", end="")
    # print("\nAltitude Heatmap:")
    # print(altitude_heatmap)


    # grid_img = env.render_grid(altitude_heatmap)
    # print(grid_img)
