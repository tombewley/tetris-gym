import gym, numpy as np, matplotlib.pyplot as plt

class TetrisEnv(gym.Env):
    """Ultra-minimal Tetris Gym environment,
    with original Nintindo scoring system and simplified game mechanics."""
    metadata = {"render.modes": ["human", "rgb_array"]}
    tetrominos = (np.array([[1,1,1,1]]),       # 0: I
                  np.array([[1,1,1],[0,0,1]]), # 1: J
                  np.array([[1,1,1],[1,0,0]]), # 2: L
                  np.array([[1,1],[1,1]]),     # 3: O
                  np.array([[0,1,1],[1,1,0]]), # 4: S
                  np.array([[1,1,1],[0,1,0]]), # 5: T
                  np.array([[1,1,0],[0,1,1]])) # 6: Z
    num_upcoming = 4 # Length of lookahead to upcoming pieces
    clear_reward = [0., 40., 100., 300., 1200.] # https://tetris.fandom.com/wiki/Scoring
    timestep_reward = 0.
    done_reward = -100.

    def __init__(self, board_shape=(20,10), vector_obs=False, render_mode=False):
        self.state_space = gym.spaces.Dict({
            "board":    gym.spaces.MultiBinary(board_shape), # Board is binary array
            "upcoming": gym.spaces.MultiDiscrete([7]*self.num_upcoming)}) # 7 possible tetromino pieces
        if vector_obs: # Flattened board and one-hot representation of upcoming
            self.observation_space = gym.spaces.MultiBinary(
                (board_shape[0]*board_shape[1]) + (7*self.num_upcoming))
        else: # Board and upcoming as-is in dictionary format
            self.observation_space = self.state_space
        self.board = self.state_space.spaces["board"].sample()
        # NOTE: Could use MultiDiscrete but for small board_shape[1] (e.g. 10) factoring is unnecessary
        self.action_space = gym.spaces.Discrete(board_shape[1] * 4) # x position, 90deg rotations
        self.vector_obs, self.render_mode = vector_obs, render_mode
        if self.render_mode: # Set up matplotlib rendering
            assert self.render_mode in self.metadata["render.modes"]
            if self.render_mode == "rgb_array": plt.switch_backend("agg")
            fig, ax = plt.subplots(figsize=(2,4))
            self.board_img = ax.imshow(self.board)
            plt.ion(); ax.axis("off"); fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.seed()

    def seed(self, seed=None):
        self.state_space.seed(seed)
        self.action_space.seed(seed)

    def reset(self):
        self.board.fill(0) # Clear board
        self.upcoming = self.state_space.spaces["upcoming"].sample() # Sample a list of upcoming pieces
        return self.obs()

    def step(self, action):
        assert action in self.action_space
        piece = np.rot90(self.tetrominos[self.upcoming[0]], action%4) # Select and rotate next tetromino
        pos_x = min(action//4, self.board.shape[1]-piece.shape[1]) # Clip x position to board size
        slice_x = np.s_[pos_x:pos_x+piece.shape[1]] # Range of x positions covered by piece
        max_y = self.board.shape[0]+1-piece.shape[0] # Maximum possible y position for piece 
        done = False
        for pos_y in range(max_y): # Iterate *downwards*
            slice_y = np.s_[pos_y:pos_y+piece.shape[0]] # Range of y positions covered by piece
            if np.logical_and(self.board[slice_y,slice_x], piece).any(): # Cannot place here
                if pos_y == 0: done = True # Top of board reached without placing
                break
            slice_y_prev = slice_y
        if done: num_full = 0
        else:
            self.board[slice_y_prev,slice_x] += piece # Place piece on board
            full = self.board.sum(axis=1) == self.board.shape[1] # Identify full rows to clear
            num_full = full.sum()
            if num_full: self.board[num_full:] = self.board[~full] # Clear all full rows simultaneously
        self.upcoming[:-1] = self.upcoming[1:] # Remove piece from upcoming...
        self.upcoming[-1] = self.state_space.spaces["upcoming"].sample()[-1] # ...and sample a replacement
        reward = self.clear_reward[num_full] + self.timestep_reward + (done * self.done_reward)
        return self.obs(), reward, done, {}

    def obs(self):
        if self.vector_obs:
            upcoming_one_hot = np.zeros(7*self.num_upcoming, dtype=np.int8)
            upcoming_one_hot[self.upcoming + np.arange(self.num_upcoming*7, step=7)] = 1
            return np.hstack((self.board.flatten(), upcoming_one_hot))
        else:
            return {"board": self.board, "upcoming": self.upcoming}

    def render(self, mode="human", pause=1e-6):
        assert mode == self.render_mode, f"Render mode is {self.render_mode}, so cannot use {mode}"
        self.board_img.set_data(self.board)
        self.board_img._axes.set_title(str(self.upcoming))
        if self.render_mode == "human": plt.pause(pause)
        elif self.render_mode == "rgb_array": # https://stackoverflow.com/a/7821917
            self.board_img.figure.canvas.draw()
            data = np.fromstring(self.board_img.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            return data.reshape(self.board_img.figure.canvas.get_width_height()[::-1] + (3,))
