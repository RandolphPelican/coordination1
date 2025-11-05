import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def create_position_heatmap(logger, world_size=15):
    """Create heatmap showing where agents spent most time"""
    heatmap = np.zeros((world_size, world_size))
    
    for agent_id, history in logger.agent_histories.items():
        for state in history:
            pos = state['position']
            if 0 <= pos[0] < world_size and 0 <= pos[1] < world_size:
                heatmap[pos[1], pos[0]] += 1
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        colorscale='YlOrRd',
        colorbar=dict(title='Visits'),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Visits: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Agent Position Heatmap (Density)',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        width=600,
        height=600,
        template='plotly_white'
    )
    
    return fig


def create_trajectory_visualization(logger, world_size=15, max_agents=5):
    """Create visualization showing agent trajectories"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    agent_ids = list(logger.agent_histories.keys())[:max_agents]
    
    for idx, agent_id in enumerate(agent_ids):
        trajectory = logger.get_agent_trajectory(agent_id)
        
        if trajectory:
            steps, positions = zip(*trajectory)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                name=agent_id,
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6),
                hovertemplate=f'{agent_id}<br>Step: %{{customdata}}<br>X: %{{x}}<br>Y: %{{y}}<extra></extra>',
                customdata=steps
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode='markers',
                marker=dict(size=12, color=colors[idx % len(colors)], symbol='star'),
                showlegend=False,
                hovertemplate=f'{agent_id} Start<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'Agent Trajectories (First {max_agents} agents)',
        xaxis=dict(range=[-0.5, world_size-0.5], title='X Position'),
        yaxis=dict(range=[-0.5, world_size-0.5], title='Y Position'),
        width=700,
        height=700,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def create_decision_heatmap(logger, world_size=15):
    """Create heatmap showing decision types by location"""
    decision_maps = {
        'seek_food': np.zeros((world_size, world_size)),
        'avoid_danger': np.zeros((world_size, world_size)),
        'random': np.zeros((world_size, world_size))
    }
    
    move_df = logger.get_movement_dataframe()
    
    if not move_df.empty:
        for _, row in move_df.iterrows():
            pos = row['old_pos']
            decision = row['decision_basis']
            
            if 0 <= pos[0] < world_size and 0 <= pos[1] < world_size:
                if decision in decision_maps:
                    decision_maps[decision][pos[1], pos[0]] += 1
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Food-Seeking Decisions', 'Danger-Avoiding Decisions', 'Random Decisions')
    )
    
    fig.add_trace(
        go.Heatmap(z=decision_maps['seek_food'], colorscale='Greens', showscale=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(z=decision_maps['avoid_danger'], colorscale='Reds', showscale=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(z=decision_maps['random'], colorscale='Greys', showscale=True,
                  colorbar=dict(title='Count')),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Decision Type Heatmaps by Location',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_agent_efficiency_comparison(logger):
    """Compare individual agent performance"""
    coord_df = logger.get_coordination_dataframe()
    
    if coord_df.empty:
        return None
    
    agent_stats = []
    
    for agent_id in logger.agent_histories.keys():
        food_collected = len(coord_df[
            (coord_df['agent_id'] == agent_id) & 
            (coord_df['event_type'] == 'food_collected')
        ])
        
        dangers_hit = len(coord_df[
            (coord_df['agent_id'] == agent_id) & 
            (coord_df['event_type'] == 'danger_hit')
        ])
        
        efficiency = food_collected - dangers_hit
        
        agent_stats.append({
            'agent': agent_id,
            'food': food_collected,
            'dangers': dangers_hit,
            'efficiency': efficiency
        })
    
    df = pd.DataFrame(agent_stats).sort_values('efficiency', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['agent'],
        x=df['food'],
        name='Food Collected',
        orientation='h',
        marker_color='#27ae60'
    ))
    
    fig.add_trace(go.Bar(
        y=df['agent'],
        x=[-d for d in df['dangers']],
        name='Dangers Hit',
        orientation='h',
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title='Agent Performance Comparison',
        xaxis_title='Count',
        yaxis_title='Agent',
        barmode='relative',
        template='plotly_white',
        height=max(400, len(df) * 30)
    )
    
    return fig


def create_message_flow_network(logger, step=None):
    """Create network visualization of message flow between agents"""
    msg_df = logger.get_message_dataframe()
    
    if msg_df.empty:
        return None
    
    if step is not None:
        msg_df = msg_df[msg_df['step'] == step]
    
    delivered_msgs = msg_df[msg_df['delivered']]
    
    if delivered_msgs.empty:
        return None
    
    agent_ids = list(logger.agent_histories.keys())
    n_agents = len(agent_ids)
    
    angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
    positions = {
        agent_id: (np.cos(angle), np.sin(angle))
        for agent_id, angle in zip(agent_ids, angles)
    }
    
    fig = go.Figure()
    
    for sender_id in delivered_msgs['sender_id'].unique():
        sender_pos = positions.get(sender_id)
        if sender_pos is None:
            continue
            
        for receiver_id in agent_ids:
            if receiver_id != sender_id:
                receiver_pos = positions[receiver_id]
                
                fig.add_trace(go.Scatter(
                    x=[sender_pos[0], receiver_pos[0]],
                    y=[sender_pos[1], receiver_pos[1]],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.3)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    for agent_id, pos in positions.items():
        msg_count = len(delivered_msgs[delivered_msgs['sender_id'] == agent_id])
        
        fig.add_trace(go.Scatter(
            x=[pos[0]],
            y=[pos[1]],
            mode='markers+text',
            marker=dict(
                size=20 + msg_count * 5,
                color='#3498db',
                line=dict(color='white', width=2)
            ),
            text=[agent_id.split('_')[1]],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            name=agent_id,
            hovertext=f'{agent_id}<br>Messages sent: {msg_count}',
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title='Message Flow Network',
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        width=600,
        height=600,
        template='plotly_white'
    )
    
    return fig


def create_coordination_timeline(logger):
    """Create timeline showing coordination metrics over time"""
    step_df = logger.get_step_summary_dataframe()
    move_df = logger.get_movement_dataframe()
    
    if step_df.empty or move_df.empty:
        return None
    
    coord_by_step = move_df.groupby('step')['moved_toward_goal'].mean()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Bandwidth Utilization',
            'Message Delivery Rate',
            'Coordination Rate'
        ),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=step_df['step'],
            y=step_df['bandwidth_used'],
            fill='tozeroy',
            name='Bandwidth Used',
            line=dict(color='#3498db')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=step_df['step'],
            y=step_df['bandwidth_efficiency'],
            fill='tozeroy',
            name='Delivery Rate',
            line=dict(color='#e67e22')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=coord_by_step.index,
            y=coord_by_step.values,
            fill='tozeroy',
            name='Coordination',
            line=dict(color='#2ecc71')
        ),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text='Step', row=3, col=1)
    fig.update_yaxes(title_text='Bits', row=1, col=1)
    fig.update_yaxes(title_text='Rate', row=2, col=1, tickformat='.0%')
    fig.update_yaxes(title_text='Rate', row=3, col=1, tickformat='.0%')
    
    fig.update_layout(
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_exploration_coverage_map(logger, world_size=15):
    """Create map showing exploration coverage over time"""
    visited = np.zeros((world_size, world_size))
    
    for agent_id, history in logger.agent_histories.items():
        for state in history:
            pos = state['position']
            if 0 <= pos[0] < world_size and 0 <= pos[1] < world_size:
                visited[pos[1], pos[0]] = 1
    
    coverage_rate = visited.sum() / (world_size * world_size)
    
    fig = go.Figure(data=go.Heatmap(
        z=visited,
        colorscale=[[0, '#ecf0f1'], [1, '#27ae60']],
        showscale=False,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Visited: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Exploration Coverage: {coverage_rate:.1%} of world visited',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        width=600,
        height=600,
        template='plotly_white'
    )
    
    return fig
