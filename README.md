# Atari setup
We will first train an Atari game called DemonAttack, https://gym.openai.com/envs/DemonAttack-v0

## Install gym environment
Install atari environment for gym with: 
> pip install gym[atari]

## Try a sample program 
    import gym
    env = gym.make('DemonAttack-v0')
    EPISODES = 1000 
    ACTION_NOTHING = 0
    ACTION_FIRE = 1
    ACTION_RIGHT = 2
    ACTION_LEFT = 3
    ACTION_RIGHT_AND_FIRE = 4
    ACTION_LEFT_AND_FIRE = 5
    
    state = env.reset()
    state = env.step(0)
    env.render()
    for e in range(EPISODES):
         print(e)
         print(state)
         env.step(ACTION_FIRE)
         env.render()

Now with this, you will have a running environment which will render the game, and keep pressing the FIRE button on every step.

![](http://nepascene.com/wp-content/themes/patterns/timthumb.php?src=http%3A%2F%2Fnepascene.com%2Fwp-content%2Fuploads%2F2015%2F01%2Fdemon-attack-atari-2600.png&q=90&w=650&zc=1)

Now with that, as you can see, you have 6 different actions that you can perform on the environment. In the next step, we will "observe" the state of the game (in form of array of pixels) and train our model to perform the best "action" on every "step".

[.. more steps to follow]

# References

* [Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) - A very nice tutorial for reinforcement learning 
* [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/) - A tutorial with excellent code snippets
* [Deep Atari](https://github.com/bhaktipriya/Atari) - A python implementation of a DRL algorithm that learns to play Atari games using the raw pixels as its input
* [Andrew Ng's Machine Learning course](https://www.coursera.org/learn/machine-learning) - A free, online introductory course to machine learning
* [Udacity Machine Learning](https://www.youtube.com/watch?v=Ki2iHgKxRBo) - Introductory video for a machine learning course on Udacity, which contains three sections for reinforcement learning
* [Demystifying Deep Reinforcement Learning](https://ai.intel.com/demystifying-deep-reinforcement-learning/)
* [David Silver's DeepMind RL course](https://www.youtube.com/playlist?list=PLbWDNovNB5mqFBgq7i3MY6Ui4zudcvNFJ)
# Videos

* [Google DeepMind's Deep Q-learning playing Atari Breakout](https://www.youtube.com/watch?v=V1eYniJ0Rnk)
