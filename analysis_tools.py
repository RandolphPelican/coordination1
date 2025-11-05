import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from simulation_core import SimulationEnvironment

def run_bandwidth_sweep(bandwidths, n_agents=8, n_episodes=5, n_steps=30, 
                       world_size=15, num_food=10, num_dangers=5, seed=None):
    """Run experiments across different bandwidth settings"""
    results = []
    
    for bw in bandwidths:
        bandwidth_results = {
            'bandwidth': bw,
            'efficiencies': [],
            'coordination_rates': [],
            'message_delivery_rates': [],
            'food_collected': [],
            'dangers_hit': []
        }
        
        for episode in range(n_episodes):
            episode_seed = seed + episode if seed is not None else None
            
            env = SimulationEnvironment(
                world_size=world_size,
                num_agents=n_agents,
                num_food=num_food,
                num_dangers=num_dangers,
                bandwidth_bits=bw
            )
            env.initialize(seed=episode_seed)
            stats = env.run_episode(num_steps=n_steps)
            
            efficiency = stats.get('net_efficiency', 0)
            coord_rate = stats.get('coordination_rate', 0)
            msg_delivery = stats.get('message_delivery_rate', 0)
            food = stats.get('food_collected', 0)
            dangers = stats.get('dangers_hit', 0)
            
            bandwidth_results['efficiencies'].append(efficiency)
            bandwidth_results['coordination_rates'].append(coord_rate)
            bandwidth_results['message_delivery_rates'].append(msg_delivery)
            bandwidth_results['food_collected'].append(food)
            bandwidth_results['dangers_hit'].append(dangers)
        
        bandwidth_results['mean_efficiency'] = np.mean(bandwidth_results['efficiencies'])
        bandwidth_results['std_efficiency'] = np.std(bandwidth_results['efficiencies'])
        bandwidth_results['mean_coordination'] = np.mean(bandwidth_results['coordination_rates'])
        bandwidth_results['std_coordination'] = np.std(bandwidth_results['coordination_rates'])
        bandwidth_results['mean_msg_delivery'] = np.mean(bandwidth_results['message_delivery_rates'])
        
        results.append(bandwidth_results)
    
    return results


def detect_inverted_u_curve(results, min_improvement=1.5):
    """Detect if results show an inverted U-curve pattern"""
    if len(results) < 3:
        return False, "Need at least 3 bandwidth levels"
    
    efficiencies = [r['mean_efficiency'] for r in results]
    peak_idx = np.argmax(efficiencies)
    peak_value = efficiencies[peak_idx]
    
    has_left_increase = peak_idx > 0 and (peak_value - efficiencies[0]) > min_improvement
    has_right_decrease = peak_idx < len(efficiencies) - 1 and (peak_value - efficiencies[-1]) > min_improvement
    
    is_inverted_u = has_left_increase and has_right_decrease
    
    if is_inverted_u:
        message = f"Inverted U-curve detected! Peak at bandwidth {results[peak_idx]['bandwidth']} bits"
    else:
        message = "No clear inverted U-curve pattern"
    
    return is_inverted_u, message, peak_idx


def run_causal_ablation_test(bandwidth_constrained=1000, bandwidth_unconstrained=100000,
                             n_agents=12, n_episodes=15, n_steps=40, seed=None):
    """Run A-B-C causal ablation test"""
    
    def run_phase(bandwidth, phase_name, seed_offset):
        phase_results = []
        
        for episode in range(n_episodes):
            episode_seed = seed + seed_offset + episode if seed is not None else None
            
            env = SimulationEnvironment(
                world_size=15,
                num_agents=n_agents,
                num_food=12,
                num_dangers=6,
                bandwidth_bits=bandwidth
            )
            env.initialize(seed=episode_seed)
            stats = env.run_episode(num_steps=n_steps)
            
            phase_results.append({
                'efficiency': stats.get('net_efficiency', 0),
                'coordination': stats.get('coordination_rate', 0),
                'msg_delivery': stats.get('message_delivery_rate', 0)
            })
        
        return {
            'phase': phase_name,
            'bandwidth': bandwidth,
            'mean_efficiency': np.mean([r['efficiency'] for r in phase_results]),
            'std_efficiency': np.std([r['efficiency'] for r in phase_results]),
            'mean_coordination': np.mean([r['coordination'] for r in phase_results]),
            'efficiencies': [r['efficiency'] for r in phase_results]
        }
    
    phase_a = run_phase(bandwidth_constrained, 'A: Constrained', 0)
    phase_b = run_phase(bandwidth_unconstrained, 'B: Unconstrained', 1000)
    phase_c = run_phase(bandwidth_constrained, 'C: Re-Constrained', 2000)
    
    a_eff = phase_a['mean_efficiency']
    b_eff = phase_b['mean_efficiency']
    c_eff = phase_c['mean_efficiency']
    
    drop = a_eff - b_eff
    recovery = c_eff - b_eff
    consistency = abs(a_eff - c_eff)
    
    test_removal_hurts = drop > 1.5
    test_restoration_helps = recovery > 1.5
    test_consistent = consistency < 2.5
    
    tests_passed = sum([test_removal_hurts, test_restoration_helps, test_consistent])
    causality_confirmed = tests_passed >= 2
    
    t_stat, p_value = stats.ttest_ind(phase_a['efficiencies'], phase_b['efficiencies'])
    
    return {
        'phase_a': phase_a,
        'phase_b': phase_b,
        'phase_c': phase_c,
        'drop': drop,
        'recovery': recovery,
        'consistency': consistency,
        'test_removal_hurts': test_removal_hurts,
        'test_restoration_helps': test_restoration_helps,
        'test_consistent': test_consistent,
        'tests_passed': tests_passed,
        'causality_confirmed': causality_confirmed,
        't_statistic': t_stat,
        'p_value': p_value
    }


def create_inverted_u_plot(results):
    """Create interactive plot showing inverted U-curve"""
    bandwidths = [r['bandwidth'] for r in results]
    efficiencies = [r['mean_efficiency'] for r in results]
    errors = [r['std_efficiency'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=bandwidths,
        y=efficiencies,
        mode='lines+markers',
        name='Efficiency',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10),
        error_y=dict(
            type='data',
            array=errors,
            visible=True
        )
    ))
    
    fig.update_layout(
        title='Inverted U-Curve: Bandwidth vs. Coordination Efficiency',
        xaxis_title='Bandwidth (bits)',
        yaxis_title='Net Efficiency (Food - Dangers)',
        xaxis_type='log',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_phase_comparison_plot(causal_results):
    """Create bar chart comparing A-B-C phases"""
    phases = ['Phase A\n(Constrained)', 'Phase B\n(Unconstrained)', 'Phase C\n(Re-Constrained)']
    efficiencies = [
        causal_results['phase_a']['mean_efficiency'],
        causal_results['phase_b']['mean_efficiency'],
        causal_results['phase_c']['mean_efficiency']
    ]
    errors = [
        causal_results['phase_a']['std_efficiency'],
        causal_results['phase_b']['std_efficiency'],
        causal_results['phase_c']['std_efficiency']
    ]
    
    colors = ['#2ecc71', '#e74c3c', '#2ecc71']
    
    fig = go.Figure(data=[
        go.Bar(
            x=phases,
            y=efficiencies,
            error_y=dict(type='data', array=errors),
            marker_color=colors,
            text=[f'{e:.1f}' for e in efficiencies],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Causal Ablation Test: A-B-C Phase Comparison',
        yaxis_title='Net Efficiency',
        template='plotly_white',
        height=450,
        showlegend=False
    )
    
    return fig


def create_grid_visualization(state, agent_colors=None):
    """Create grid visualization of current environment state"""
    world_size = state['world_size']
    
    grid = np.zeros((world_size, world_size))
    
    for pos in state['food']:
        if 0 <= pos[0] < world_size and 0 <= pos[1] < world_size:
            grid[pos[1], pos[0]] = 1
    
    for pos in state['dangers']:
        if 0 <= pos[0] < world_size and 0 <= pos[1] < world_size:
            grid[pos[1], pos[0]] = -1
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=[
            [0, '#ffe6e6'],
            [0.5, '#f0f0f0'],
            [1, '#e6ffe6']
        ],
        showscale=False,
        hoverinfo='skip'
    ))
    
    agent_positions = list(state['agents'].values())
    if agent_positions:
        agent_x = [pos[0] for pos in agent_positions]
        agent_y = [pos[1] for pos in agent_positions]
        agent_names = list(state['agents'].keys())
        
        fig.add_trace(go.Scatter(
            x=agent_x,
            y=agent_y,
            mode='markers+text',
            marker=dict(size=15, color='#3498db', line=dict(width=2, color='white')),
            text=[name.split('_')[1] for name in agent_names],
            textfont=dict(color='white', size=8),
            name='Agents',
            hovertext=agent_names,
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f'Environment Grid (Step {state["step"]})',
        xaxis=dict(range=[-0.5, world_size-0.5], showgrid=True, dtick=1),
        yaxis=dict(range=[-0.5, world_size-0.5], showgrid=True, dtick=1),
        width=600,
        height=600,
        template='plotly_white'
    )
    
    return fig
