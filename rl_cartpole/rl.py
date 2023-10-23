#%% Imports
import gym
from collections import deque
import numpy as np
import tensorflow as tf

from pydantic import BaseModel, Field, validator
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow import keras

#%% Classes
# Environment class
class Env():
    def __init__(self, env_name: str="CartPole-v0"):
        self.env = gym.make(env_name)
        self.shape_state = list(self.env.observation_space.shape)
        self.n_action = self.env.action_space.n
    
    def plot_env(self):
        PIL.Image.fromarray(self.env.render())

# Experience class (experience can be learnt externally)
class Exp():
    def __init__(self, buffer_len:int=2000, seed: int=None):
        self.buffer_len = buffer_len
        self.replay_buffer = deque(maxlen=self.buffer_len)
        self.rng = np.random.default_rng(seed=seed)
        pass

    def add_to_exp(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_exp(self, batch_size):
        replay_indices = self.rng.integers(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in replay_indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[f] for experience in batch])
            for f in range(5)
        ] # 5 fields in an experience: state, action, reward, next_state, done
        return states, actions, rewards, next_states, dones

# Learning pydantic model
class LearningModel(BaseModel):
    initial_epsilon: float = 1
    min_epsilon: float = 0.01
    epsilon: float = Field(initial_epsilon, description="Should be equal to initial_epsilon")
    decay_rate: float = 1/500
    batch_size: int = 32
    discount_factor: float = 0.95
    model: keras.Model = Field(..., description="A Keras model")
    learning_rate: float = 0.001
    optimizer: keras.optimizers.Optimizer = Field(keras.optimizers.legacy.Adam(learning_rate=learning_rate))
    loss_fn: keras.losses.Loss = Field(keras.losses.mean_squared_error)

    class Config:
        arbitrary_types_allowed = True

    @validator("epsilon")
    def check_initial_eps(cls, value, values):
        if value != values.get("initial_epsilon"):
            raise ValueError("epsilon must be equal to initial_epsilon")
        return value


# Actor class
class Actor():
    '''
    Args:
    env: Environment of agent
    exp: Experience available to actor

    '''
    def __init__(self, env: Env, exp: Exp, 
                 action_model: LearningModel,
                 target_model: LearningModel=None,

                 seed: int=None
        ):
        self.env = env
        self.current_state = self.env.env.reset()
        self.step_index = 0
        self.episode_index = 0

        self.exp = exp
        self.action_model = action_model
        self.target_model = target_model

        self.rng = np.random.default_rng(seed=seed)
        pass
    
    def start_new_episode(self, first_episode: bool=False):
        self.current_state = self.env.env.reset()
        self.step_index = 0
        if first_episode:
            self.episode_index = 0
        else:
            self.episode_index += 1


    def update_epsilon(self, model):
        self.action_model.epsilon = max(
            model.initial_epsilon - self.episode_index * model.decay_rate, 
            model.min_epsilon
        )

    def get_action(self):
        self.update_epsilon(self.action_model)
        if self.rng.random() < self.action_model.epsilon:
            return self.rng.integers(self.env.n_action)
        else:
            Q_values = self.action_model.model.predict(self.current_state[np.newaxis], verbose=0)
            return np.argmax(Q_values[0])
        

    def take_step(self):
        self.last_action = self.get_action()
        self.next_state, self.last_reward, self.done, _ = self.env.env.step(self.last_action)
        self.exp.add_to_exp(self.current_state, 
                            self.last_action, 
                            self.last_reward,
                            self.next_state, 
                            self.done)
        self.current_state = self.next_state
    

    def update_model(self):
        training_exp = self.exp.sample_exp(self.action_model.batch_size) # should we use batch size from action model???
        states, actions, rewards, next_states, dones = training_exp

        next_Q_values = self.action_model.model.predict(next_states, verbose=0)
        best_actions_in_next_states = np.argmax(next_Q_values, axis=1)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        # tf.reduce_sum(
        #     next_Q_values * tf.one_hot(best_actions_in_next_states, self.env.n_action),
        #     axis=1, keepdims=True
        # )
        target_Q_values = rewards + (1-dones)*self.action_model.discount_factor*max_next_Q_values

        actions_mask = tf.one_hot(actions, self.env.n_action)
        with tf.GradientTape() as tape:
            current_Q_values = self.action_model.model(states)
            action_Q_values = tf.reduce_sum(current_Q_values * actions_mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.action_model.loss_fn(target_Q_values, action_Q_values))
        gradients = tape.gradient(loss, self.action_model.model.trainable_variables)
        self.action_model.optimizer.apply_gradients(
            zip(gradients, self.action_model.model.trainable_variables)
        )
        
# Simulation episode class
class SimEpisodes():
    def __init__(
            self, actor: Actor, n_max_steps: int=200,
            n_episodes:int=600, min_training_len: int=50
    ):
        self.actor = actor
        self.n_max_steps = n_max_steps
        self.n_episodes = n_episodes
        self.min_training_len = min_training_len
        if min_training_len < 1.5 * actor.action_model.batch_size:
            self.min_training_len = int(1.5 * actor.action_model.batch_size)


    def simulate(self):
        total_rewards = []
        for sim in range(self.n_episodes):
            actor.start_new_episode(first_episode=sim==0)
            reward = 0
            for step in range(self.n_max_steps):
                actor.take_step()
                reward += actor.last_reward
                if actor.done:
                    break
            total_rewards += [reward]
            print(sim, actor.action_model.epsilon, total_rewards[-1])
            if sim > self.min_training_len:
                actor.update_model()
        
        return total_rewards

#%% Main
if __name__=="__main__":
    # Initialise environment
    cartpole_env = Env()

    # Define learning models
    action_model = LearningModel(
        model = keras.models.Sequential([
            keras.layers.Dense(32, activation='elu', input_shape=cartpole_env.shape_state),
            keras.layers.Dense(32, activation='elu'),
            keras.layers.Dense(cartpole_env.n_action)
        ]),
        learning_rate=0.001,
        decay_rate=1/200
        # ,loss_fn = keras.losses.mean_squared_error
    )
    target_model = action_model

    # Define experience
    exp = Exp()

    # Finally define actor using these
    actor = Actor(
        env=cartpole_env, exp=exp, 
        action_model=action_model,
        target_model=target_model,
        seed=42
    )

    simulation = SimEpisodes(actor=actor)
    total_rewards = simulation.simulate()
    plt.plot(range(len(total_rewards)), total_rewards)





    

        




# %%