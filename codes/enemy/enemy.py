class Enemy:
    def __init__(self, X, G):
        self.X = X
        self.G = G
        self.cur_ind = 0

    def has_moves(self):
        return self.cur_ind < self.X.shape[0]

    def next_x(self):
        return self.X[self.cur_ind]

    def get_g(self):
        g = self.G[self.cur_ind]
        self.cur_ind += 1
        return g
