import numpy as np

"""
Класс, представляющий собой одну кинематическую пару
"""


class Robot:
    parts = []
    penaltiesMin = None
    penaltiesMax = None
    scores = 100

    def __init__(self, parts):
        self.parts = parts
        self.penaltiesMin = [(p.borderMin) for p in self.parts]
        self.penaltiesMax = [(p.borderMax) for p in self.parts]

    """
    Получить значение штрафа для данных обобщенных координат
    """

    def penalty(self, Q, W1=1, W2=1):

        # lambda (t): (200 * exp(-t)) if t > 200 else (400 * exp(-t))
        reduce_to_nil = lambda n: 0 if n > 0 else np.abs(n)

        subtract = np.subtract(Q, self.penaltiesMin)
        np_subtract = np.subtract(self.penaltiesMax, Q)
        return W1 * np.sum(list(map(reduce_to_nil, subtract))) + W2 * np.sum(
            list(map(reduce_to_nil, np_subtract))
        )

    """
    Получить координаты схвата (конечного звена)
    """

    def getXYZ(self, Q):
        return self.getXYZPair(Q, len(self.parts))[:3]

    """
    Получить координаты конкретной пары 
    """

    def getXYZPair(self, Q, pair):

        resultMatrix = np.eye(4, dtype=np.float32)

        for i, p in enumerate(self.parts):

            if i == pair:
                break

            resultMatrix = np.matmul(resultMatrix, p.getMatrix(Q[i]))

        xyz1 = np.matmul(resultMatrix, [[0], [0], [0], [1]])

        return xyz1

    """
    Массив координат всех пар (для построения графика)
    """

    def getPairPoints(self, Q):

        result = []

        for i, p in enumerate(self.parts):
            pairXYZ = self.getXYZPair(Q, i)
            result.append([pairXYZ[0], pairXYZ[1], pairXYZ[2]])

        return result


class KinematicPart:
    s = 0
    a = 0
    alpha = 0

    borderMin = 0
    borderMax = 0

    def __init__(self, s, a, alpha, bmin, bmax):
        self.s = s
        self.a = a
        self.alpha = alpha
        self.borderMin = bmin
        self.borderMax = bmax

    def getMatrix(self, q):
        return [
            [
                np.cos(q),
                -np.sin(q) * np.cos(self.alpha),
                np.sin(q) * np.sin(self.alpha),
                self.a * np.cos(q),
            ],
            [
                np.sin(q),
                np.cos(q) * np.cos(self.alpha),
                -np.cos(q) * np.sin(self.alpha),
                self.a * np.sin(q),
            ],
            [0, np.sin(self.alpha), np.cos(self.alpha), self.s],
            [0, 0, 0, 1],
        ]


r = np.pi / 180.0

#
Z1 = KinematicPart(400, 180, np.pi / 2, bmin=-185 * r, bmax=185 * r)
Z2 = KinematicPart(135, 600, 180 * r, bmin=180 * r, bmax=270 * r)
Z3 = KinematicPart(135, 120, -90 * r, bmin=-90 * r, bmax=360 * r)
Z4 = KinematicPart(620, 0, 90 * r, bmin=180 * r, bmax=180 * r)
Z5 = KinematicPart(0, 0, -90 * r, bmin=-5 * r, bmax=15 * r)
Z6 = KinematicPart(115, 0, 0, bmin=-5 * r, bmax=15 * r)
# RevoluteJoint
# revoluteJoint1 = CreateRevoluteJoint(400.0, 180.0, 90.0, 0.0, 60.0);
# RevoluteJoint
# revoluteJoint2 = CreateRevoluteJoint(135.0, 600.0, 180.0, 0.0, 60.0);
# RevoluteJoint
# revoluteJoint3 = CreateRevoluteJoint(135.0, 120.0, -90.0, 0.0, 60.0);
# RevoluteJoint
# revoluteJoint4 = CreateRevoluteJoint(620.0, 0.0, 90.0, 0.0, 60.0);
# RevoluteJoint
# revoluteJoint5 = CreateRevoluteJoint(0.0, 0.0, -90.0, 0.0, 60.0);
# RevoluteJoint
# revoluteJoint6 = CreateRevoluteJoint(115.0, 0.0, 0.0, 0.0, 60.0);
parts = [Z1, Z2, Z3, Z4, Z5, Z6]

RV = Robot(parts)


Q01 = 0 * r
Q12 = 0 * r
Q23 = 0 * r
Q34 = 0 * r
Q45 = 0 * r
Q56 = 0 * r


Q0 = [Q01, Q12, Q23, Q34, Q45, Q56]

if __name__ == "__main__":
    print(RV.getXYZ(Q0))
