class AbstractOnlineLearner:
    def __init__(self, experts):
        self.experts = experts

    def start_game(self, enemy):
        total_loss = 0.0
        while enemy.has_moves():
            x = enemy.next_x()

            g_hat = self.respond(x)
            g = enemy.get_g()
            loss = self.calculate_loss(g, g_hat)
            total_loss += loss

            self.learn(g)

        return total_loss

    def respond(self, x):
        raise NotImplementedError

    def learn(self, g):
        raise NotImplementedError

    def calculate_loss(self, g, g_hat):
        return int(g != g_hat)
