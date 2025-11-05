import pandas as pd
from collections import defaultdict
from datetime import datetime

class SimulationLogger:
    """Comprehensive logging system for agent coordination simulations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all logs for a new simulation run"""
        self.message_logs = []
        self.movement_logs = []
        self.coordination_logs = []
        self.event_logs = []
        self.step_summaries = []
        self.agent_histories = defaultdict(list)
        
    def log_message_event(self, step, sender_id, message_type, position, importance, delivered):
        """Log a message generation and delivery event"""
        self.message_logs.append({
            'step': step,
            'sender_id': sender_id,
            'message_type': message_type,
            'position': position,
            'importance': importance,
            'delivered': delivered,
            'timestamp': datetime.now()
        })
    
    def log_movement(self, step, agent_id, old_pos, new_pos, decision_basis, moved_toward_goal):
        """Log agent movement and decision-making"""
        self.movement_logs.append({
            'step': step,
            'agent_id': agent_id,
            'old_pos': old_pos,
            'new_pos': new_pos,
            'decision_basis': decision_basis,
            'moved_toward_goal': moved_toward_goal,
            'distance_traveled': abs(new_pos[0] - old_pos[0]) + abs(new_pos[1] - old_pos[1])
        })
    
    def log_coordination_event(self, step, agent_id, event_type, position, value=None):
        """Log coordination events (food collected, danger hit, etc.)"""
        self.coordination_logs.append({
            'step': step,
            'agent_id': agent_id,
            'event_type': event_type,
            'position': position,
            'value': value
        })
    
    def log_step_summary(self, step, num_agents, num_food, num_dangers, 
                        messages_delivered, messages_dropped, bandwidth_used):
        """Log summary statistics for each timestep"""
        self.step_summaries.append({
            'step': step,
            'num_agents': num_agents,
            'num_food': num_food,
            'num_dangers': num_dangers,
            'messages_delivered': messages_delivered,
            'messages_dropped': messages_dropped,
            'bandwidth_used': bandwidth_used,
            'bandwidth_efficiency': messages_delivered / max(messages_delivered + messages_dropped, 1)
        })
    
    def log_agent_state(self, step, agent_id, position, num_food_alerts, 
                       num_danger_warnings, success_rate):
        """Log individual agent state"""
        self.agent_histories[agent_id].append({
            'step': step,
            'position': position,
            'num_food_alerts': num_food_alerts,
            'num_danger_warnings': num_danger_warnings,
            'success_rate': success_rate
        })
    
    def get_message_dataframe(self):
        """Get message logs as pandas DataFrame"""
        if not self.message_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.message_logs)
    
    def get_movement_dataframe(self):
        """Get movement logs as pandas DataFrame"""
        if not self.movement_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.movement_logs)
    
    def get_coordination_dataframe(self):
        """Get coordination event logs as pandas DataFrame"""
        if not self.coordination_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.coordination_logs)
    
    def get_step_summary_dataframe(self):
        """Get step summaries as pandas DataFrame"""
        if not self.step_summaries:
            return pd.DataFrame()
        return pd.DataFrame(self.step_summaries)
    
    def get_agent_trajectory(self, agent_id):
        """Get position trajectory for a specific agent"""
        if agent_id not in self.agent_histories:
            return []
        return [(state['step'], state['position']) for state in self.agent_histories[agent_id]]
    
    def get_summary_statistics(self):
        """Get overall summary statistics"""
        msg_df = self.get_message_dataframe()
        coord_df = self.get_coordination_dataframe()
        move_df = self.get_movement_dataframe()
        
        stats = {}
        
        if not msg_df.empty:
            stats['total_messages_generated'] = len(msg_df)
            stats['total_messages_delivered'] = msg_df['delivered'].sum()
            stats['message_delivery_rate'] = msg_df['delivered'].mean()
            stats['avg_message_importance'] = msg_df['importance'].mean()
        
        if not coord_df.empty:
            stats['food_collected'] = (coord_df['event_type'] == 'food_collected').sum()
            stats['dangers_hit'] = (coord_df['event_type'] == 'danger_hit').sum()
            stats['net_efficiency'] = stats.get('food_collected', 0) - stats.get('dangers_hit', 0)
        
        if not move_df.empty:
            stats['coordinated_moves'] = move_df['moved_toward_goal'].sum()
            stats['total_moves'] = len(move_df)
            stats['coordination_rate'] = move_df['moved_toward_goal'].mean() if len(move_df) > 0 else 0
        
        return stats
