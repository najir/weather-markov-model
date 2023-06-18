import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distrobutions

initDistro = tfd.categorical(probs=[0.8,0.2])
transDistro = tfd.categorical(probs=[[0.7,0.3],[0.2,0.8]])
obsDistro = tfd.Normal(loc=[0.0,15.0], scale=[5.0,10.0])

markovModel = tfd.HiddenMarkovModel(
    initial_distrobution = initDistro,
    transition_distrobution = transDistro,
    observation_distrobution = obsDistro,
    num_steps = 7
)

weatherMean = markovModel.mean()

with tf.compat.v1.session as sess:
    print(weatherMean.numpy())

tempData = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

#Given our list of data points for tempature we can predicate the most likely change in states
markovModel.posterior_mode(tempData)


