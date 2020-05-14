import matplotlib.pyplot as plt
import numpy as np
import sys

#implementing value iteration for the gambler's problem

class GamblerPolIter:
    _max_cash = 100
    _min_cash = 0
    _heads_prob = 0.4
    _disc_fact = 0.9

    value = np.zeros(_max_cash + 1, dtype=float)    #include state zero cash
    policy = value.copy().astype(int)

    def set_values(self, val):
        self.value = np.array(val)

    def set_policy(self, pol):
        self.policy = np.array(pol)

    def expected_reward(self, state, action):
        """
        state  : It's the amount of actual cash at each step
        action : The amount of cash bet on heads at each step
        """

        rw = 0  # reward

        # terminal states
        if state == 0:
            return 0
        if state == 100:
            return 0

        # there is one discrete random variable which determine the probability distribution of the reward and next state
        # the variable is the coin result

        for val, Prob in enumerate([self._heads_prob, 1-self._heads_prob]):

            if val == 1:
                new_state = max(min(state - action, self._max_cash), 0)
            elif val == 0:
                new_state = max(min(state + action, self._max_cash), 0)

            # reward is 1 for the final positive state and zero otherwise
            rew = 1 if new_state >= self._max_cash else 0

            # Bellman's equation
            rw += Prob * (rew + self._disc_fact * self.value[new_state])

        return rw

    # initial value of eps_param
    _eps_param = 1e-3

    def policy_iteration(self):

        # here policy_evaluation has a static variable eps_param whose values decreases over time
        eps_param = self._eps_param

        self._eps_param /= 10

        while True:
            delta_param = 0

            for s in range(1, self.value.size-1):
                old_val = self.value[s]
                # max action allowed is actual cash amount (s)
                exp_rw = [self.expected_reward(s, a) for a in range(s+1)]
                self.value[s] = np.max(exp_rw)

                delta_param = max(delta_param, abs(self.value[s] - old_val))

                print('.', end='')
                sys.stdout.flush()
            print(delta_param)
            sys.stdout.flush()

            if delta_param < eps_param:
                break
        for s in range(1, self.policy.size-1):
            exp_rw = [self.expected_reward(s, a) for a in range(s+1)]
            act_chos = np.argmax(exp_rw)
            self.policy[s] = np.argmax(exp_rw)


    def save_value(self):
        with open("value_list.txt", "w") as file:
            print("saving file...")
            file.write(str(self.value) + "\n")

    def save_policy(self):
        with open("policy_list.txt", "w") as file:
            print("saving file...")
            file.write(str(self.policy) + "\n")

def read_from_file(filename, start=0):
    data = []
    with open(filename, "r") as file:
        [next(file) for _ in range(start-1)]
        str_line = file.readline()
        while str_line:
            str_line = str_line.strip("[]\n")
            for str_char in str_line.split(" "):
                try:
                    data.append(float(str_char))
                except ValueError:
                    pass
            str_line = file.readline()
    return data

def plot_result():
    policy_data = read_from_file("policy_list.txt")
    value_data = read_from_file("value_list.txt")
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(range(101), value_data, "b-")
    plt.xlabel("capital")
    plt.ylabel("values")
    plt.ylim([0,1])
    plt.xlim([0,99])
    plt.subplot(1,2,2)
    plt.plot(range(101), policy_data, "k-", ds="steps")
    plt.xlabel("capital")
    plt.ylabel("stakes")

    plt.show()

def run_problem():
    prob = GamblerPolIter()
    prob.policy_iteration()
    prob.save_value()
    prob.save_policy()

if __name__ == "__main__":
    run_problem()
    plot_result()