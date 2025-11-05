import numpy as np
from collections import deque
import random
from simulation_logger import SimulationLogger

class CoordinationAgent:
    """Enhanced agent with comprehensive logging capabilities"""
    
    def __init__(self, agent_id, world_size=15, vision_radius=3, logger=None):
        self.agent_id = agent_id
        self.world_size = world_size
        self.vision_radius = vision_radius
        self.position = (random.randint(0, world_size-1), random.randint(0, world_size-1))
        self.message_success_history = deque(maxlen=10)
        self.logger = logger
        
        self.received_food_alerts = []
        self.received_danger_warnings = []
        self.total_food_collected = 0
        self.total_dangers_hit = 0
    
    def generate_message(self, environment_state):
        """Generate messages about what agent can SEE"""
        messages = []
        
        for food_pos in environment_state.get('food_locations', []):
            dist = abs(food_pos[0] - self.position[0]) + abs(food_pos[1] - self.position[1])
            if dist <= self.vision_radius:
                messages.append({
                    'type': 'food_alert',
                    'position': food_pos,
                    'importance': 0.8,
                    'agent_id': self.agent_id
                })
        
        for danger_pos in environment_state.get('danger_locations', []):
            dist = abs(danger_pos[0] - self.position[0]) + abs(danger_pos[1] - self.position[1])
            if dist <= self.vision_radius:
                messages.append({
                    'type': 'danger_warning',
                    'position': danger_pos,
                    'importance': 0.9,
                    'agent_id': self.agent_id
                })
        
        if self.message_success_history:
            success_rate = np.mean(list(self.message_success_history))
            if success_rate < 0.3 and random.random() < 0.5:
                return None
        
        if messages:
            messages.sort(key=lambda m: m['importance'], reverse=True)
            return messages[0]
        return None
    
    def receive_message(self, message):
        """Store received messages for decision-making"""
        if message['type'] == 'food_alert':
            self.received_food_alerts.append(message['position'])
        elif message['type'] == 'danger_warning':
            self.received_danger_warnings.append(message['position'])
    
    def decide_movement(self):
        """Make movement decision based on received messages"""
        decision_basis = 'random'
        
        if self.received_danger_warnings:
            decision_basis = 'avoid_danger'
            nearest_danger = min(
                self.received_danger_warnings,
                key=lambda d: abs(d[0] - self.position[0]) + abs(d[1] - self.position[1])
            )
            
            dx = self.position[0] - nearest_danger[0]
            dy = self.position[1] - nearest_danger[1]
            
            if abs(dx) > abs(dy):
                return (1 if dx > 0 else -1, 0), decision_basis
            else:
                return (0, 1 if dy > 0 else -1), decision_basis
        
        if self.received_food_alerts:
            decision_basis = 'seek_food'
            nearest_food = min(
                self.received_food_alerts,
                key=lambda f: abs(f[0] - self.position[0]) + abs(f[1] - self.position[1])
            )
            
            dx = nearest_food[0] - self.position[0]
            dy = nearest_food[1] - self.position[1]
            
            if abs(dx) > abs(dy):
                return (1 if dx > 0 else -1, 0), decision_basis
            else:
                return (0, 1 if dy > 0 else -1), decision_basis
        
        decision_basis = 'random'
        return random.choice([(0,1), (1,0), (0,-1), (-1,0)]), decision_basis
    
    def clear_messages(self):
        """Clear messages at start of new timestep"""
        self.received_food_alerts = []
        self.received_danger_warnings = []
    
    def record_message_success(self, was_delivered):
        self.message_success_history.append(1 if was_delivered else 0)
    
    def get_success_rate(self):
        """Get current message success rate"""
        if not self.message_success_history:
            return 0.0
        return np.mean(list(self.message_success_history))


class CoordinationMAC:
    """Enhanced MAC with comprehensive logging"""
    
    def __init__(self, agents, bandwidth_bits=1000, logger=None):
        self.agents = agents
        self.bandwidth_bits = bandwidth_bits
        self.logger = logger
        self.message_size_bits = 100
    
    def coordinate(self, environment_state, current_step=0):
        """Coordinate agents with bandwidth constraints and logging"""
        for agent in self.agents.values():
            agent.clear_messages()
        
        candidate_messages = []
        for agent_id, agent in self.agents.items():
            message = agent.generate_message(environment_state)
            if message:
                importance = message.get('importance', 0.5)
                candidate_messages.append((importance, self.message_size_bits, agent_id, message))
        
        candidate_messages.sort(reverse=True, key=lambda x: x[0])
        
        delivered_messages = []
        dropped_messages = []
        used_bits = 0
        
        for importance, message_bits, sender_id, message in candidate_messages:
            if used_bits + message_bits <= self.bandwidth_bits:
                for receiver_id, receiver in self.agents.items():
                    if receiver_id != sender_id:
                        receiver.receive_message(message)
                
                delivered_messages.append((sender_id, message))
                used_bits += message_bits
                self.agents[sender_id].record_message_success(True)
                
                if self.logger:
                    self.logger.log_message_event(
                        current_step, sender_id, message['type'], 
                        message['position'], importance, True
                    )
            else:
                dropped_messages.append((sender_id, message))
                self.agents[sender_id].record_message_success(False)
                
                if self.logger:
                    self.logger.log_message_event(
                        current_step, sender_id, message['type'], 
                        message['position'], importance, False
                    )
        
        return {
            'delivered': len(delivered_messages),
            'dropped': len(dropped_messages),
            'candidates': len(candidate_messages),
            'bandwidth_used': used_bits,
            'bandwidth_available': self.bandwidth_bits
        }


class SimulationEnvironment:
    """Enhanced simulation environment with logging"""
    
    def __init__(self, world_size=15, num_agents=8, num_food=10, num_dangers=5, 
                 bandwidth_bits=1000, vision_radius=3, logger=None):
        self.world_size = world_size
        self.num_agents = num_agents
        self.num_food = num_food
        self.num_dangers = num_dangers
        self.bandwidth_bits = bandwidth_bits
        self.vision_radius = vision_radius
        self.logger = logger or SimulationLogger()
        
        self.agents = {}
        self.mac = None
        self.food_locations = []
        self.danger_locations = []
        self.current_step = 0
        
    def initialize(self, seed=None):
        """Initialize or reset the environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.logger.reset()
        self.current_step = 0
        
        self.agents = {
            f"agent_{i}": CoordinationAgent(
                f"agent_{i}", self.world_size, self.vision_radius, self.logger
            ) for i in range(self.num_agents)
        }
        
        self.mac = CoordinationMAC(self.agents, self.bandwidth_bits, self.logger)
        
        self.food_locations = [
            (random.randint(0, self.world_size-1), random.randint(0, self.world_size-1)) 
            for _ in range(self.num_food)
        ]
        
        self.danger_locations = [
            (random.randint(0, self.world_size-1), random.randint(0, self.world_size-1)) 
            for _ in range(self.num_dangers)
        ]
    
    def step(self):
        """Execute one simulation step"""
        env_state = {
            'time': self.current_step,
            'food_locations': self.food_locations.copy(),
            'danger_locations': self.danger_locations.copy(),
            'world_size': self.world_size
        }
        
        coord_result = self.mac.coordinate(env_state, self.current_step)
        
        for agent_id, agent in self.agents.items():
            old_pos = agent.position
            (dx, dy), decision_basis = agent.decide_movement()
            
            new_x = max(0, min(self.world_size-1, agent.position[0] + dx))
            new_y = max(0, min(self.world_size-1, agent.position[1] + dy))
            agent.position = (new_x, new_y)
            
            moved_toward_goal = self._check_goal_movement(old_pos, agent.position)
            
            if self.logger:
                self.logger.log_movement(
                    self.current_step, agent_id, old_pos, agent.position,
                    decision_basis, moved_toward_goal
                )
                
                self.logger.log_agent_state(
                    self.current_step, agent_id, agent.position,
                    len(agent.received_food_alerts),
                    len(agent.received_danger_warnings),
                    agent.get_success_rate()
                )
            
            if agent.position in self.food_locations:
                self.food_locations.remove(agent.position)
                agent.total_food_collected += 1
                if self.logger:
                    self.logger.log_coordination_event(
                        self.current_step, agent_id, 'food_collected', 
                        agent.position, value=1
                    )
            
            if agent.position in self.danger_locations:
                agent.total_dangers_hit += 1
                if self.logger:
                    self.logger.log_coordination_event(
                        self.current_step, agent_id, 'danger_hit', 
                        agent.position, value=-1
                    )
        
        if self.logger:
            self.logger.log_step_summary(
                self.current_step, self.num_agents, len(self.food_locations),
                len(self.danger_locations), coord_result['delivered'],
                coord_result['dropped'], coord_result['bandwidth_used']
            )
        
        self.current_step += 1
        
        return coord_result
    
    def _check_goal_movement(self, old_pos, new_pos):
        """Check if agent moved toward food or away from danger"""
        if not self.food_locations and not self.danger_locations:
            return False
        
        moved_toward_goal = False
        
        if self.food_locations:
            old_dist = min(abs(old_pos[0]-f[0])+abs(old_pos[1]-f[1]) for f in self.food_locations)
            new_dist = min(abs(new_pos[0]-f[0])+abs(new_pos[1]-f[1]) for f in self.food_locations)
            if new_dist < old_dist:
                moved_toward_goal = True
        
        if self.danger_locations:
            old_dist = min(abs(old_pos[0]-d[0])+abs(old_pos[1]-d[1]) for d in self.danger_locations)
            new_dist = min(abs(new_pos[0]-d[0])+abs(new_pos[1]-d[1]) for d in self.danger_locations)
            if new_dist > old_dist:
                moved_toward_goal = True
        
        return moved_toward_goal
    
    def get_state(self):
        """Get current environment state for visualization"""
        return {
            'step': self.current_step,
            'agents': {aid: agent.position for aid, agent in self.agents.items()},
            'food': self.food_locations.copy(),
            'dangers': self.danger_locations.copy(),
            'world_size': self.world_size
        }
    
    def run_episode(self, num_steps=30):
        """Run a complete episode"""
        for _ in range(num_steps):
            self.step()
        
        return self.logger.get_summary_statistics()
