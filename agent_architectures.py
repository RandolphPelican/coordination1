import numpy as np
from abc import ABC, abstractmethod
import random

class AgentStrategy(ABC):
    """Base class for agent decision strategies"""
    
    def __init__(self, agent_id, vision_radius):
        self.agent_id = agent_id
        self.vision_radius = vision_radius
        self.memory = []
    
    @abstractmethod
    def decide_action(self, state, messages):
        """Decide action based on current state and messages"""
        pass
    
    @abstractmethod
    def get_strategy_name(self):
        """Return strategy name"""
        pass
    
    def _get_visible_objects(self, state):
        """Get objects within vision radius"""
        pos = state['agents'][self.agent_id]['position']
        
        visible_food = []
        visible_dangers = []
        
        for food_pos in state['food']:
            dist = np.linalg.norm(np.array(pos) - np.array(food_pos))
            if dist <= self.vision_radius:
                visible_food.append(food_pos)
        
        for danger_pos in state['dangers']:
            dist = np.linalg.norm(np.array(pos) - np.array(danger_pos))
            if dist <= self.vision_radius:
                visible_dangers.append(danger_pos)
        
        return visible_food, visible_dangers
    
    def _calculate_direction(self, from_pos, to_pos):
        """Calculate direction vector from one position to another"""
        diff = np.array(to_pos) - np.array(from_pos)
        if np.linalg.norm(diff) == 0:
            return np.array([0, 0])
        return diff / np.linalg.norm(diff)


class GreedyStrategy(AgentStrategy):
    """Aggressive strategy: Always chase nearest food, ignore dangers unless very close"""
    
    def get_strategy_name(self):
        return "Greedy"
    
    def decide_action(self, state, messages):
        pos = np.array(state['agents'][self.agent_id]['position'])
        visible_food, visible_dangers = self._get_visible_objects(state)
        
        immediate_dangers = [d for d in visible_dangers 
                            if np.linalg.norm(np.array(d) - pos) < 1.5]
        
        if immediate_dangers:
            danger_pos = np.array(immediate_dangers[0])
            escape_dir = pos - danger_pos
            if np.linalg.norm(escape_dir) > 0:
                escape_dir = escape_dir / np.linalg.norm(escape_dir)
            return {
                'move': escape_dir.tolist(),
                'broadcast': {'type': 'danger_alert', 'position': danger_pos.tolist()}
            }
        
        if visible_food:
            nearest_food = min(visible_food, key=lambda f: np.linalg.norm(np.array(f) - pos))
            direction = self._calculate_direction(pos, nearest_food)
            return {
                'move': direction.tolist(),
                'broadcast': {'type': 'food_found', 'position': nearest_food}
            }
        
        for msg in messages:
            if msg.get('type') == 'food_found':
                food_pos = np.array(msg['position'])
                direction = self._calculate_direction(pos, food_pos)
                return {'move': direction.tolist(), 'broadcast': None}
        
        random_dir = np.random.randn(2)
        random_dir = random_dir / np.linalg.norm(random_dir)
        return {'move': random_dir.tolist(), 'broadcast': None}


class CautiousStrategy(AgentStrategy):
    """Defensive strategy: Avoid dangers first, then seek food"""
    
    def get_strategy_name(self):
        return "Cautious"
    
    def decide_action(self, state, messages):
        pos = np.array(state['agents'][self.agent_id]['position'])
        visible_food, visible_dangers = self._get_visible_objects(state)
        
        if visible_dangers:
            danger_pos = np.array(visible_dangers[0])
            escape_dir = pos - danger_pos
            if np.linalg.norm(escape_dir) > 0:
                escape_dir = escape_dir / np.linalg.norm(escape_dir)
            return {
                'move': escape_dir.tolist(),
                'broadcast': {'type': 'danger_alert', 'position': danger_pos.tolist()}
            }
        
        safe_food = []
        for food_pos in visible_food:
            food_arr = np.array(food_pos)
            is_safe = True
            for danger_pos in state['dangers']:
                if np.linalg.norm(food_arr - np.array(danger_pos)) < 2.0:
                    is_safe = False
                    break
            if is_safe:
                safe_food.append(food_pos)
        
        if safe_food:
            nearest_safe_food = min(safe_food, key=lambda f: np.linalg.norm(np.array(f) - pos))
            direction = self._calculate_direction(pos, nearest_safe_food)
            return {
                'move': direction.tolist(),
                'broadcast': {'type': 'safe_food', 'position': nearest_safe_food}
            }
        
        for msg in messages:
            if msg.get('type') == 'safe_food':
                food_pos = np.array(msg['position'])
                direction = self._calculate_direction(pos, food_pos)
                return {'move': direction.tolist(), 'broadcast': None}
        
        random_dir = np.random.randn(2)
        random_dir = random_dir / np.linalg.norm(random_dir)
        return {'move': random_dir.tolist(), 'broadcast': None}


class BalancedStrategy(AgentStrategy):
    """Balanced strategy: Weight both food seeking and danger avoidance"""
    
    def get_strategy_name(self):
        return "Balanced"
    
    def decide_action(self, state, messages):
        pos = np.array(state['agents'][self.agent_id]['position'])
        visible_food, visible_dangers = self._get_visible_objects(state)
        
        danger_weight = 2.0
        food_weight = 1.0
        
        net_direction = np.array([0.0, 0.0])
        
        for danger_pos in visible_dangers:
            danger_arr = np.array(danger_pos)
            dist = np.linalg.norm(danger_arr - pos)
            if dist > 0:
                repulsion = (pos - danger_arr) / dist
                strength = danger_weight / max(dist, 0.5)
                net_direction += repulsion * strength
        
        for food_pos in visible_food:
            food_arr = np.array(food_pos)
            dist = np.linalg.norm(food_arr - pos)
            if dist > 0:
                attraction = (food_arr - pos) / dist
                strength = food_weight / max(dist, 0.5)
                net_direction += attraction * strength
        
        for msg in messages:
            if msg.get('type') == 'food_found':
                food_pos = np.array(msg['position'])
                dist = np.linalg.norm(food_pos - pos)
                if dist > 0:
                    attraction = (food_pos - pos) / dist
                    strength = 0.3 / max(dist, 1.0)
                    net_direction += attraction * strength
            elif msg.get('type') == 'danger_alert':
                danger_pos = np.array(msg['position'])
                dist = np.linalg.norm(danger_pos - pos)
                if dist > 0:
                    repulsion = (pos - danger_pos) / dist
                    strength = 0.5 / max(dist, 1.0)
                    net_direction += repulsion * strength
        
        if np.linalg.norm(net_direction) > 0:
            net_direction = net_direction / np.linalg.norm(net_direction)
        else:
            net_direction = np.random.randn(2)
            net_direction = net_direction / np.linalg.norm(net_direction)
        
        broadcast_msg = None
        if visible_food:
            broadcast_msg = {'type': 'food_found', 'position': visible_food[0]}
        elif visible_dangers:
            broadcast_msg = {'type': 'danger_alert', 'position': visible_dangers[0]}
        
        return {
            'move': net_direction.tolist(),
            'broadcast': broadcast_msg
        }


class ExplorerStrategy(AgentStrategy):
    """Explorer strategy: Prioritize exploring unseen areas"""
    
    def __init__(self, agent_id, vision_radius):
        super().__init__(agent_id, vision_radius)
        self.visited_positions = set()
        self.exploration_target = None
    
    def get_strategy_name(self):
        return "Explorer"
    
    def decide_action(self, state, messages):
        pos = np.array(state['agents'][self.agent_id]['position'])
        visible_food, visible_dangers = self._get_visible_objects(state)
        
        self.visited_positions.add(tuple(np.round(pos).astype(int)))
        
        close_dangers = [d for d in visible_dangers 
                        if np.linalg.norm(np.array(d) - pos) < 2.0]
        
        if close_dangers:
            danger_pos = np.array(close_dangers[0])
            escape_dir = pos - danger_pos
            if np.linalg.norm(escape_dir) > 0:
                escape_dir = escape_dir / np.linalg.norm(escape_dir)
            return {
                'move': escape_dir.tolist(),
                'broadcast': {'type': 'danger_alert', 'position': danger_pos.tolist()}
            }
        
        if visible_food and random.random() < 0.5:
            nearest_food = min(visible_food, key=lambda f: np.linalg.norm(np.array(f) - pos))
            direction = self._calculate_direction(pos, nearest_food)
            return {
                'move': direction.tolist(),
                'broadcast': {'type': 'food_found', 'position': nearest_food}
            }
        
        world_size = state.get('world_size', 15)
        
        if self.exploration_target is None or np.linalg.norm(pos - self.exploration_target) < 1.0:
            self.exploration_target = np.array([
                random.uniform(0, world_size),
                random.uniform(0, world_size)
            ])
        
        direction = self._calculate_direction(pos, self.exploration_target)
        
        return {
            'move': direction.tolist(),
            'broadcast': {'type': 'exploring', 'coverage': len(self.visited_positions)}
        }


class CooperativeStrategy(AgentStrategy):
    """Cooperative strategy: Heavily prioritize team coordination via messages"""
    
    def get_strategy_name(self):
        return "Cooperative"
    
    def decide_action(self, state, messages):
        pos = np.array(state['agents'][self.agent_id]['position'])
        visible_food, visible_dangers = self._get_visible_objects(state)
        
        for msg in messages:
            if msg.get('type') == 'danger_alert':
                danger_pos = np.array(msg['position'])
                dist = np.linalg.norm(danger_pos - pos)
                if dist < 3.0:
                    escape_dir = pos - danger_pos
                    if np.linalg.norm(escape_dir) > 0:
                        escape_dir = escape_dir / np.linalg.norm(escape_dir)
                    return {'move': escape_dir.tolist(), 'broadcast': None}
        
        if visible_dangers:
            danger_pos = np.array(visible_dangers[0])
            escape_dir = pos - danger_pos
            if np.linalg.norm(escape_dir) > 0:
                escape_dir = escape_dir / np.linalg.norm(escape_dir)
            return {
                'move': escape_dir.tolist(),
                'broadcast': {'type': 'danger_alert', 'position': danger_pos.tolist(), 'sender': self.agent_id}
            }
        
        for msg in messages:
            if msg.get('type') == 'food_found':
                if msg.get('sender') != self.agent_id:
                    food_pos = np.array(msg['position'])
                    direction = self._calculate_direction(pos, food_pos)
                    return {'move': direction.tolist(), 'broadcast': None}
        
        if visible_food:
            nearest_food = min(visible_food, key=lambda f: np.linalg.norm(np.array(f) - pos))
            direction = self._calculate_direction(pos, nearest_food)
            return {
                'move': direction.tolist(),
                'broadcast': {'type': 'food_found', 'position': nearest_food, 'sender': self.agent_id}
            }
        
        random_dir = np.random.randn(2)
        random_dir = random_dir / np.linalg.norm(random_dir)
        return {
            'move': random_dir.tolist(),
            'broadcast': {'type': 'searching', 'sender': self.agent_id}
        }


class AgentArchitectureManager:
    """Manage different agent architectures and strategies"""
    
    AVAILABLE_STRATEGIES = {
        'greedy': GreedyStrategy,
        'cautious': CautiousStrategy,
        'balanced': BalancedStrategy,
        'explorer': ExplorerStrategy,
        'cooperative': CooperativeStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name, agent_id, vision_radius):
        """Create a strategy instance"""
        strategy_class = cls.AVAILABLE_STRATEGIES.get(strategy_name.lower(), BalancedStrategy)
        return strategy_class(agent_id, vision_radius)
    
    @classmethod
    def get_strategy_descriptions(cls):
        """Get descriptions of all available strategies"""
        return {
            'greedy': 'Aggressively chases food, only avoids immediate dangers',
            'cautious': 'Prioritizes safety, only seeks food when safe',
            'balanced': 'Balances food seeking with danger avoidance using weighted vectors',
            'explorer': 'Focuses on exploring new areas while opportunistically collecting food',
            'cooperative': 'Heavily relies on team communication and shared information'
        }
    
    @classmethod
    def get_available_strategies(cls):
        """Get list of available strategy names"""
        return list(cls.AVAILABLE_STRATEGIES.keys())
