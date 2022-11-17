import tensorflow as tf
import numpy as np

'''
Класс, представляющий собой одну кинематическую пару
'''


class KinematicPart:
    s = tf.constant(0, dtype=tf.float32)
    a = tf.constant(0, dtype=tf.float32)
    alpha = tf.constant(0, dtype=tf.float32)

    borderMin = tf.constant(0, dtype=tf.float32)
    borderMax = tf.constant(0, dtype=tf.float32)

    def __init__(self, s, a, alpha, bmin, bmax):
        self.s = tf.constant(s, dtype=tf.float32)
        self.a = tf.constant(a, dtype=tf.float32)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.borderMin = tf.constant(bmin, dtype=tf.float32)
        self.borderMax = tf.constant(bmax, dtype=tf.float32)

    def getMatrix(self, q):
        return [
            [tf.cos(q), -tf.sin(q) * tf.cos(self.alpha), tf.sin(q) * tf.sin(self.alpha), self.a * tf.cos(q)],
            [tf.sin(q), tf.cos(q) * tf.cos(self.alpha), -tf.cos(q) * tf.sin(self.alpha), self.a * tf.sin(q)],
            [0, tf.sin(self.alpha), tf.cos(self.alpha), self.s],
            [0, 0, 0, 1]
        ]


'''
Класс робота, состоящего из пар
'''


class Robot:
    parts = []
    penaltiesMin = None
    penaltiesMax = None

    def __init__(self, parts):
        self.parts = parts
        self.penaltiesMin = [(p.borderMin) for p in self.parts]
        self.penaltiesMax = [(p.borderMax) for p in self.parts]

    '''
    Получить значение штрафа для данных обобщенных координат
    '''

    def penalty(self, Q, W1=1, W2=1):

        reduce_to_nil = lambda n: tf.cond(n > 0,
                                          lambda: tf.constant(0, dtype=tf.float32), lambda: tf.abs(n))

        return W1 * tf.reduce_sum(
            tf.map_fn(reduce_to_nil, tf.subtract(Q, self.penaltiesMin))
        ) + W2 * tf.reduce_sum(tf.map_fn(reduce_to_nil, tf.subtract(self.penaltiesMax, Q)))

    '''
    Получить координаты схвата (конечного звена)
    '''

    def getXYZ(self, Q):
        return self.getXYZPair(Q, len(self.parts))[:3]

    '''
    Получить координаты конкретной пары 
    '''

    def getXYZPair(self, Q, pair):

        resultMatrix = tf.eye(4, dtype=tf.float32)

        for i, p in enumerate(self.parts):

            if i == pair:
                break

            resultMatrix = tf.matmul(resultMatrix, p.getMatrix(Q[i]))

        xyz1 = tf.matmul(resultMatrix, tf.constant([[0], [0], [0], [1]], dtype=tf.float32))

        return xyz1

    '''
    Массив координат всех пар (для построения графика)
    '''

    def getPairPoints(self, Q):

        result = []

        for i, p in enumerate(self.parts):
            pairXYZ = self.getXYZPair(Q, i)
            result.append([pairXYZ[0], pairXYZ[1], pairXYZ[2]])

        return result


r = np.pi / 180.0

#
Z1 = KinematicPart(400, 180, np.pi / 2, bmin=-185 * r, bmax=185 * r)
Z2 = KinematicPart(135, 600, 180*r, bmin=180 * r, bmax=270 * r)
Z3 = KinematicPart(135, 120, -90*r, bmin=-90 * r, bmax=360 * r)
Z4 = KinematicPart(620, 0, 90*r, bmin=180 * r, bmax=180 * r)
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


Q01 = tf.Variable(0 * r, dtype=tf.float32)
Q12 = tf.Variable(0 * r, dtype=tf.float32)
Q23 = tf.Variable(0 * r, dtype=tf.float32)
Q34 = tf.Variable(0 * r, dtype=tf.float32, trainable=False)
Q45 = tf.Variable(0 * r, dtype=tf.float32)
Q56 = tf.Variable(0 * r, dtype=tf.float32)


Q0 = [Q01, Q12, Q23, Q34, Q45, Q56]

print(RV.getXYZ(Q0).numpy())

