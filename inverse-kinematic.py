import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from kinematicPart import KinematicPart
from robot import Robot

trajectory = [
    [[260], [0], [540]],
    [[259], [0], [535]],
    [[259], [10], [530]],
    [[259], [15], [525]],
    [[255], [10], [520]],
    [[250], [0], [515]],
]

radian = np.pi / 180.0

Z1 = KinematicPart(300, 0, np.pi / 2, bmin=-185 * radian, bmax=185 * radian)
Z2 = KinematicPart(0, 250, 0, bmin=50 * radian, bmax=270 * radian)
Z3 = KinematicPart(0, 160, 0, bmin=-360 * radian, bmax=360 * radian)
Z4 = KinematicPart(0, 0, np.pi / 2, bmin=-180 * radian, bmax=180 * radian)
Z5 = KinematicPart(0, 104.9, np.pi / 2, bmin=0 * radian, bmax=0 * radian)

parts = [Z1, Z2, Z3, Z4, Z5]

RV = Robot(parts)

Q01 = tf.Variable(0.00093 * radian, dtype=tf.float32)
Q12 = tf.Variable(90 * radian, dtype=tf.float32)
Q23 = tf.Variable(270 * radian, dtype=tf.float32)
Q34 = tf.Variable(0 * radian, dtype=tf.float32)
Q45 = tf.Variable(0 * radian, dtype=tf.float32, trainable=False)

Q0 = [Q01, Q12, Q23, Q34, Q45]

target = tf.Variable([[260], [0], [540]], dtype=tf.float32)

learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.5, decay_steps=10, decay_rate=0.96, staircase=False
)


def loss_function(Q0):
    xyz = RV.getXYZ(Q0)
    penalty = RV.penalty(Q0, 9999, 9999)
    return tf.reduce_sum(tf.sqrt(tf.pow(target - xyz, 2))) + penalty


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)

loss = 99999

obobq1 = []
obobq2 = []
obobq3 = []
obobq4 = []
obobq5 = []

for val in trajectory:

    target = tf.Variable([val[0], val[1], val[2]], dtype=tf.float32)
    loss = loss_function(Q0)
    while loss > 0.04:
        with tf.GradientTape() as tape:
            y = loss_function(Q0)
        grads = tape.gradient(y, [Q01, Q12, Q23, Q34])
        grads_and_vars = zip(grads, [Q01, Q12, Q23, Q34])

        print("#######################")
        print(
            "Q01 = {:.5f}, Q02 = {:.5f}, Q03 = {:.5f},  Q04 = {:.5f}, Q05 = {:.5f}, grads = {:.5f}".format(
                Q01.numpy() / radian,
                Q12.numpy() / radian,
                Q23.numpy() / radian,
                Q34.numpy() / radian,
                Q45.numpy() / radian,
                y.numpy(),
            )
        )
        optimizer.apply_gradients(grads_and_vars)
        print("target: = ")
        print(target.numpy())
        print("xyz: = ")
        print(RV.getXYZ(Q0).numpy())
        loss = y.numpy()

    obobq1.append(Q01.numpy() / radian)
    obobq2.append(Q12.numpy() / radian)
    obobq3.append(Q23.numpy() / radian)
    obobq4.append(Q34.numpy() / radian)
    obobq5.append(Q45.numpy() / radian)

print(obobq1)
print(obobq2)
print(obobq3)
print(obobq4)
print(obobq5)

fig = plt.figure()

ax = fig.gca()

ax.set_ylabel("Значение обобщенной координаты")
ax.set_xlabel("Точка")
ax.grid(color="black", linestyle="-", linewidth=0.5)
ax.plot([10, 20, 30, 40, 50, 60], obobq1, color="m")
plt.show()
