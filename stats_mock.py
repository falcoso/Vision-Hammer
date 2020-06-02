import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.close('all')

class Unit:
    def __init__(self, models, p, rapid_fire=False):
        self.models = models
        self.p = p
        self.rapid_fire = rapid_fire

    @property
    def shots(self):
        shots = self.models
        if self.rapid_fire:
            shots *= 2
        return shots

    def attack(self, other):
        """PMF of self attacking other unit."""
        pmf = stats.binom.pmf(np.arange(self.shots+1), self.shots, self.p)
        if self.shots > other.models:
            pmf2 = pmf[:other.models+1]
            pmf2[-1] += np.sum(pmf[other.models+1:])
            pmf = pmf2

        return pmf

    def conditional_attack(self, other):
        """
        Conditional PMF of self attacking other given the damage already done
        by other.
        """
        n = np.arange(max([self.models-other.shots,0]), self.models+1)
        if self.rapid_fire:
            n *=2

        model_range = np.arange(self.models+1)
        cond_pmf = np.array([stats.binom.pmf(model_range, i, self.p) for i in n])
        return cond_pmf

    def predicted_attack(self, other):
        """
        Marginal PMF of potential return attack of self attacking other, after
        other has attacked self.
        """
        cond_pmf = self.conditional_attack(other)
        pmf = other.attack(self)
        pmf = pmf[::-1] @ cond_pmf # reverse order of attack pmf
        if self.shots > other.models:
            pmf2 = pmf[:other.models+1]
            pmf2[-1] += np.sum(pmf[other.models+1:])
            pmf = pmf2

        return pmf

    def threat(self, other):
        """Expected damage self will deal against other."""
        pmf = self.attack(other)
        return np.arange(len(pmf)) @ pmf

    def conditional_threat(self, other):
        """
        Expected damage self will deal against other in response to other
        attacking self.
        """
        pmf = self.predicted_attack(other)
        return np.arange(len(pmf)) @ pmf

ps= np.array([0.5, 0.25])
x = Unit(5, ps[0])
y1 = Unit(3, ps[0])
y2 = Unit(10, ps[0])
print("Initial Threat")
print("\tX to Y1: \t{}".format(x.threat(y1)))
print("\tX to Y2: \t{}".format(x.threat(y2)))
print("\tY1 to X: \t{}".format(y1.threat(x)))
print("\tY2 to X: \t{}".format(y2.threat(x)))

print("Action 1")
x.rapid_fire = True
y1.rapid_fire = True
y2.rapid_fire = False
print("\tX attack Y1: \t{}".format(x.threat(y1)))
print("\tY1 retaliate: \t{}".format(y1.conditional_threat(x)))
x.rapid_fire = False
print("\tX attack Y2: \t{}".format(x.threat(y2)))
print("\tY2 retaliate: \t{}".format(y2.threat(x)))

print("Action 2")
x.rapid_fire = False
y1.rapid_fire = False
y2.rapid_fire = True
y2.p = ps[1]
y1.p = ps[1]
print("\tX attack Y1: \t{}".format(x.threat(y1)))
print("\tY1 retaliate: \t{}".format(y1.threat(x)))
x.rapid_fire = True
print("\tX attack Y2: \t{}".format(x.threat(y2)))
print("\tY2 retaliate: \t{}".format(y2.conditional_threat(x)))

print("Action 3")
x.rapid_fire = False
y1.rapid_fire = False
y2.rapid_fire = False
y2.p = 0
y1.p = ps[0]
print("\tX attack Y1: \t{}".format(x.threat(y1)))
print("\tY1 retaliate: \t{}".format(y1.conditional_threat(x)))
print("\tX attack Y2: \t{}".format(x.threat(y2)))
print("\tY2 retaliate: \t{}".format(y2.conditional_threat(x)))


print("Action 4")
x.rapid_fire = False
y1.rapid_fire = False
y2.rapid_fire = True
y2.p = ps[0]
y1.p = ps[0]
print("\tX attack Y1: \t{}".format(x.threat(y1)))
print("\tY1 retaliate: \t{}".format(y1.threat(x)))
x.rapid_fire = True
print("\tX attack Y2: \t{}".format(x.threat(y2)))
print("\tY2 retaliate: \t{}".format(y2.conditional_threat(x)))
