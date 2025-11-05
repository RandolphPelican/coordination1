import numpy as np
import random
from collections import defaultdict
from agent_architectures import AgentStrategy

class QLearningStrategy(AgentStrategy):
    """Q-Learning based adaptive strategy"""
    
    def __init__(self, agent_id, vision_radius, alpha=0.1, gamma=0.9, epsilon=0.2):
        super().__init__(agent_id, vision_radius)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        self.last_state = None
        self.last_action = None
        
        self.total_reward = 0
        self.episode_rewards = []
        self.q_value_history = []
    
    def get_strategy_name(self):
        return "Q-Learning"
    
    def _discretize_state(self, state, messages):
        """Convert continuous state to discrete representation"""
        pos = np.array(state['agents'][self.agent_id]['position'])
        
        food_distances = []
        for food_pos in state['food']:
            dist = np.linalg.norm(np.array(food_pos) - pos)
            if dist <= self.vision_radius * 2:
                food_distances.append(dist)
        
        danger_distances = []
        for danger_pos in state['dangers']:
            dist = np.linalg.norm(np.array(danger_pos) - pos)
            if dist <= self.vision_radius * 2:
                danger_distances.append(dist)
        
        nearest_food = min(food_distances) if food_distances else 999
        nearest_danger = min(danger_distances) if danger_distances else 999
        
        food_bin = 0 if nearest_food == 999 else (1 if nearest_food < 2 else (2 if nearest_food < 5 else 3))
        danger_bin = 0 if nearest_danger == 999 else (1 if nearest_danger < 2 else (2 if nearest_danger < 5 else 3))
        
        msg_count = len(messages)
        msg_bin = 0 if msg_count == 0 else (1 if msg_count < 3 else 2)
        
        discrete_state = (food_bin, danger_bin, msg_bin)
        
        return discrete_state
    
    def _get_action_index(self, move_vector):
        """Convert move vector to discrete action index"""
        if abs(move_vector[0]) > abs(move_vector[1]):
            return 0 if move_vector[0] > 0 else 1
        else:
            return 2 if move_vector[1] > 0 else 3
    
    def _action_index_to_vector(self, action_idx):
        """Convert action index to move vector"""
        actions = [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ]
        return actions[action_idx]
    
    def decide_action(self, state, messages):
        """Decide action using epsilon-greedy Q-learning"""
        discrete_state = self._discretize_state(state, messages)
        
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 3)
        else:
            q_values = [self.q_table[discrete_state][a] for a in range(4)]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            action_idx = random.choice(best_actions)
        
        move_vector = self._action_index_to_vector(action_idx)
        
        self.last_state = discrete_state
        self.last_action = action_idx
        
        avg_q = np.mean([self.q_table[discrete_state][a] for a in range(4)])
        self.q_value_history.append(avg_q)
        
        broadcast_msg = None
        if len(messages) > 0 and random.random() < 0.3:
            broadcast_msg = {'type': 'learning_progress', 'q_avg': avg_q}
        
        return {
            'move': move_vector,
            'broadcast': broadcast_msg
        }
    
    def update_q_value(self, reward, new_state, messages):
        """Update Q-value based on observed reward"""
        if self.last_state is None or self.last_action is None:
            return
        
        discrete_new_state = self._discretize_state(new_state, messages)
        
        future_q_values = [self.q_table[discrete_new_state][a] for a in range(4)]
        max_future_q = max(future_q_values) if future_q_values else 0
        
        current_q = self.q_table[self.last_state][self.last_action]
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        
        self.q_table[self.last_state][self.last_action] = new_q
        
        self.total_reward += reward
        self.episode_rewards.append(reward)
    
    def get_learning_stats(self):
        """Get learning statistics"""
        avg_q = 0
        if self.q_table:
            all_q_values = []
            for state_actions in self.q_table.values():
                all_q_values.extend(state_actions.values())
            avg_q = np.mean(all_q_values) if all_q_values else 0
        
        return {
            'total_reward': self.total_reward,
            'avg_q_value': avg_q,
            'states_explored': len(self.q_table),
            'epsilon': self.epsilon,
            'recent_rewards': self.episode_rewards[-10:] if self.episode_rewards else []
        }
    
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.05):
        """Decay exploration rate over time"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


class AdaptiveLearningAgent:
    """Agent that learns and adapts using Q-learning"""
    
    def __init__(self, agent_id, vision_radius, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.agent_id = agent_id
        self.vision_radius = vision_radius
        
        self.strategy = QLearningStrategy(agent_id, vision_radius, alpha, gamma, epsilon)
        
        self.last_position = None
        self.food_collected_this_step = False
        self.danger_hit_this_step = False
    
    def decide_action(self, state, messages):
        """Decide action using Q-learning strategy"""
        return self.strategy.decide_action(state, messages)
    
    def give_feedback(self, new_state, messages, food_collected=False, danger_hit=False):
        """Provide feedback to update Q-values"""
        reward = 0
        
        if food_collected:
            reward += 10.0
        
        if danger_hit:
            reward -= 5.0
        
        pos = np.array(new_state['agents'][self.agent_id]['position'])
        if self.last_position is not None:
            old_pos = np.array(self.last_position)
            
            nearest_food_dist = 999
            for food_pos in new_state['food']:
                dist = np.linalg.norm(np.array(food_pos) - pos)
                nearest_food_dist = min(nearest_food_dist, dist)
            
            old_nearest_food = 999
            for food_pos in new_state['food']:
                dist = np.linalg.norm(np.array(food_pos) - old_pos)
                old_nearest_food = min(old_nearest_food, dist)
            
            if nearest_food_dist < old_nearest_food:
                reward += 0.5
            
            nearest_danger_dist = 999
            for danger_pos in new_state['dangers']:
                dist = np.linalg.norm(np.array(danger_pos) - pos)
                nearest_danger_dist = min(nearest_danger_dist, dist)
            
            if nearest_danger_dist < 2.0:
                reward -= 1.0
        
        self.strategy.update_q_value(reward, new_state, messages)
        
        self.last_position = tuple(pos)
    
    def get_learning_stats(self):
        """Get learning statistics from strategy"""
        return self.strategy.get_learning_stats()
    
    def decay_epsilon(self, decay_rate=0.995):
        """Decay exploration rate"""
        self.strategy.decay_epsilon(decay_rate)


class RLCoordinationSimulation:
    """Simulation environment for RL agents"""
    
    def __init__(self, world_size=15, num_agents=4, num_food=10, num_dangers=5,
                 vision_radius=3, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.world_size = world_size
        self.num_agents = num_agents
        self.num_food = num_food
        self.num_dangers = num_dangers
        self.vision_radius = vision_radius
        
        self.agents = []
        self.food_locations = []
        self.danger_locations = []
        
        for i in range(num_agents):
            agent = AdaptiveLearningAgent(i, vision_radius, alpha, gamma, epsilon)
            agent.last_position = (
                random.randint(0, world_size - 1),
                random.randint(0, world_size - 1)
            )
            self.agents.append(agent)
        
        self._reset_environment()
    
    def _reset_environment(self):
        """Reset food and danger locations"""
        self.food_locations = [
            (random.randint(0, self.world_size - 1), 
             random.randint(0, self.world_size - 1))
            for _ in range(self.num_food)
        ]
        
        self.danger_locations = [
            (random.randint(0, self.world_size - 1), 
             random.randint(0, self.world_size - 1))
            for _ in range(self.num_dangers)
        ]
    
    def run_episode(self, num_steps=50):
        """Run one training episode"""
        episode_stats = {
            'total_food': 0,
            'total_dangers': 0,
            'final_q_values': [],
            'agent_rewards': []
        }
        
        for step in range(num_steps):
            for agent in self.agents:
                state = {
                    'agents': {agent.agent_id: {'position': agent.last_position}},
                    'food': self.food_locations,
                    'dangers': self.danger_locations,
                    'world_size': self.world_size
                }
                
                action = agent.decide_action(state, [])
                move = action['move']
                
                new_pos = (
                    max(0, min(self.world_size - 1, int(agent.last_position[0] + move[0]))),
                    max(0, min(self.world_size - 1, int(agent.last_position[1] + move[1])))
                )
                
                food_collected = False
                danger_hit = False
                
                if new_pos in self.food_locations:
                    self.food_locations.remove(new_pos)
                    food_collected = True
                    episode_stats['total_food'] += 1
                
                if new_pos in self.danger_locations:
                    danger_hit = True
                    episode_stats['total_dangers'] += 1
                
                agent.last_position = new_pos
                
                new_state = {
                    'agents': {agent.agent_id: {'position': new_pos}},
                    'food': self.food_locations,
                    'dangers': self.danger_locations,
                    'world_size': self.world_size
                }
                
                agent.give_feedback(new_state, [], food_collected, danger_hit)
        
        for agent in self.agents:
            stats = agent.get_learning_stats()
            episode_stats['final_q_values'].append(stats['avg_q_value'])
            episode_stats['agent_rewards'].append(stats['total_reward'])
            agent.decay_epsilon()
        
        return episode_stats
    
    def get_all_agent_stats(self):
        """Get learning stats for all agents"""
        return [agent.get_learning_stats() for agent in self.agents]
