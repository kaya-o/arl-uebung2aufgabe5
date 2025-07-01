import sys
import gym_crawlingrobot
import gymnasium as gym
import pygame
import pickle
import numpy as np

def obs_to_number(obs, obs_max):
    return int(obs[0] * obs_max + obs[1])

def obs_to_number(obs, obs_max):
    return int(obs[0] * obs_max + obs[1])

# === Value Iteration Agent ===
def vi_agent(R, Transitions, V, obs_max, env, learn=True, render=False,
             alpha=0.5, gamma=0.95, epsilon=0.1, maxSteps=2000, episodes=10):

    num_states = obs_max ** len(env.observation_space.high)
    num_actions = env.action_space.n

    print(f"R.shape={R.shape}, Transitions.shape={Transitions.shape}, V.shape={V.shape}")
    np.set_printoptions(threshold=sys.maxsize)

    # while steps < total_steps:
    for episode in range(episodes):

        if learn:
            R.fill(0)
            Transitions.fill(-1)
            V.fill(0)


        obs, _ = env.reset()
        state = obs_to_number(obs.tolist(), obs_max)
        done = False
        step = 0
        cum_reward = 0

        # while not done:
        while not done and step < maxSteps:

            if np.random.rand() < epsilon:
                action = env.action_space.sample()

            else:
                # Wähle Aktion basierend auf aktuellem V + R + Transition
                Qs = []
                for a in range(num_actions):
                    s_prime = Transitions[state, a]
                    #print(s_prime)
                    ## [s[...], s2 [....] ]
                    if s_prime >= 0:
                        q_sa = R[state, a] + gamma * V[s_prime]
                    else:
                        q_sa = R[state, a]
                    Qs.append(q_sa)
                action = np.argmax(Qs)

            nextObs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            nextState = obs_to_number(nextObs.tolist(), obs_max)
            cum_reward += reward

            if render:
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        if event.key == pygame.K_SPACE:
                            env.robot.render_intermediate_steps = not env.robot.render_intermediate_steps

            if learn:

                R[state, action] = (1 - alpha) * R[state, action] + alpha * reward
                Transitions[state, action] = nextState

                for s in range(num_states):
                    Qs = []
                    for a in range(num_actions):
                        s_prime = Transitions[s, a]
                        if s_prime >= 0:
                            q_sa = R[s, a] + gamma * V[s_prime]
                        else:
                            q_sa = R[s, a]
                        Qs.append(q_sa)
                    V[s] = max(Qs)



            state = nextState
            step += 1

            # if steps >= total_steps and not done:
            #     break

        print(f"Episode={episode+1} took {step} steps => cumulative reward: {cum_reward:.2f}")

    if render:
        while True:
            for event in pygame.event.get():
                if event.type== pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    return
    return

# === Setup Environment ===
env = gym.make('crawlingrobot-discrete-v1', rotation_angles=5, goal_distance=10000)
obs_max = int(env.observation_space.high[0] + 1)

num_states = obs_max ** len(env.observation_space.high)
num_actions = env.action_space.n

R = np.zeros((num_states, num_actions))
Transitions = -np.ones((num_states, num_actions), dtype=int)
V = np.zeros(num_states)
vi_filename = "VI_model.pkl"

### train

vi_agent(R=R, Transitions=Transitions, V=V, obs_max=obs_max, env=env,
         gamma=0.5, epsilon=0.1, alpha=0.5, episodes=5, render=False, learn=True)

pickle.dump((R, Transitions, V), open(vi_filename, "wb"))
print("Wrote VI model to file:", vi_filename)

pygame.quit()

print("Loading VI model from file:", vi_filename)
R, Transitions, V = pickle.load(open(vi_filename, "rb"))

env = gym.make('crawlingrobot-discrete-v1', rotation_angles=5,
               window_size=(640, 480), plot_steps_per_episode=True, goal_distance=10000)

vi_agent(R=R, Transitions=Transitions, V=V, obs_max=obs_max, env=env,
         gamma=0.5, epsilon=0.0, alpha=0.5, episodes=1, render=True, learn=False)