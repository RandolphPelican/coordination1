import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from simulation_core import SimulationEnvironment
from simulation_logger import SimulationLogger
from analysis_tools import (
    run_bandwidth_sweep, detect_inverted_u_curve, run_causal_ablation_test,
    create_inverted_u_plot, create_phase_comparison_plot, create_grid_visualization
)
import json
import time

st.set_page_config(
    page_title="Multi-Agent Coordination Simulator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enhanced Multi-Agent Coordination Simulator")
st.markdown("### Investigating Constraint-Driven Emergence in Communication Networks")

if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_env' not in st.session_state:
    st.session_state.current_env = None
if 'bandwidth_results' not in st.session_state:
    st.session_state.bandwidth_results = None
if 'causal_results' not in st.session_state:
    st.session_state.causal_results = None

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Simulation", 
    "📊 Bandwidth Analysis", 
    "🔬 Causal Testing",
    "📈 Analytics Dashboard",
    "💾 Export Data"
])

with tab1:
    st.header("Interactive Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        world_size = st.slider("World Size", 10, 30, 15)
        num_agents = st.slider("Number of Agents", 2, 20, 8)
        num_food = st.slider("Food Items", 5, 30, 10)
        num_dangers = st.slider("Danger Items", 2, 15, 5)
        bandwidth_bits = st.select_slider(
            "Bandwidth (bits)",
            options=[100, 500, 1000, 2000, 5000, 10000, 50000, 100000],
            value=1000
        )
        vision_radius = st.slider("Vision Radius", 1, 10, 3)
        
        st.divider()
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔄 Initialize", use_container_width=True):
                env = SimulationEnvironment(
                    world_size=world_size,
                    num_agents=num_agents,
                    num_food=num_food,
                    num_dangers=num_dangers,
                    bandwidth_bits=bandwidth_bits,
                    vision_radius=vision_radius
                )
                env.initialize(seed=42)
                st.session_state.current_env = env
                st.success("Environment initialized!")
        
        with col_btn2:
            if st.button("▶️ Step", use_container_width=True, disabled=st.session_state.current_env is None):
                if st.session_state.current_env:
                    result = st.session_state.current_env.step()
                    st.rerun()
        
        if st.button("⏩ Run 10 Steps", use_container_width=True, disabled=st.session_state.current_env is None):
            if st.session_state.current_env:
                for _ in range(10):
                    st.session_state.current_env.step()
                st.rerun()
        
        if st.button("🏁 Run Full Episode (30 steps)", use_container_width=True, disabled=st.session_state.current_env is None):
            if st.session_state.current_env:
                with st.spinner("Running episode..."):
                    stats = st.session_state.current_env.run_episode(num_steps=30)
                    st.session_state.simulation_history.append(stats)
                st.success("Episode complete!")
                st.rerun()
    
    with col2:
        st.subheader("Environment Visualization")
        
        if st.session_state.current_env:
            state = st.session_state.current_env.get_state()
            
            fig = create_grid_visualization(state)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Legend:**")
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                st.markdown("🟦 **Agents**")
            with col_l2:
                st.markdown("🟩 **Food**")
            with col_l3:
                st.markdown("🟥 **Dangers**")
            
            stats = st.session_state.current_env.logger.get_summary_statistics()
            
            st.divider()
            st.subheader("Current Statistics")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("Step", state['step'])
            with col_s2:
                st.metric("Food Remaining", len(state['food']))
            with col_s3:
                st.metric("Food Collected", stats.get('food_collected', 0))
            with col_s4:
                st.metric("Net Efficiency", stats.get('net_efficiency', 0))
            
            col_s5, col_s6, col_s7, col_s8 = st.columns(4)
            
            with col_s5:
                st.metric("Coordination Rate", f"{stats.get('coordination_rate', 0):.1%}")
            with col_s6:
                st.metric("Msg Delivery Rate", f"{stats.get('message_delivery_rate', 0):.1%}")
            with col_s7:
                st.metric("Messages Sent", stats.get('total_messages_delivered', 0))
            with col_s8:
                st.metric("Dangers Hit", stats.get('dangers_hit', 0))
        
        else:
            st.info("Click 'Initialize' to start a simulation")

with tab2:
    st.header("Bandwidth Sweep Analysis")
    st.markdown("Test the **inverted U-curve hypothesis**: moderate bandwidth constraints lead to optimal coordination")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        
        bw_type = st.radio(
            "Bandwidth Range",
            ["Quick (3 points)", "Standard (5 points)", "Detailed (7 points)", "Custom"]
        )
        
        if bw_type == "Quick (3 points)":
            bandwidths = [100, 1000, 10000]
        elif bw_type == "Standard (5 points)":
            bandwidths = [100, 500, 1000, 5000, 10000]
        elif bw_type == "Detailed (7 points)":
            bandwidths = [100, 500, 1000, 2000, 5000, 10000, 50000]
        else:
            bw_custom = st.text_input("Enter bandwidths (comma-separated)", "100,1000,10000")
            try:
                bandwidths = [int(x.strip()) for x in bw_custom.split(',')]
            except:
                bandwidths = [100, 1000, 10000]
        
        st.write(f"Testing bandwidths: {bandwidths}")
        
        bw_agents = st.slider("Agents", 4, 16, 8, key='bw_agents')
        bw_episodes = st.slider("Episodes per bandwidth", 3, 20, 5, key='bw_episodes')
        bw_steps = st.slider("Steps per episode", 20, 50, 30, key='bw_steps')
        use_seed = st.checkbox("Use fixed seed", value=True)
        seed_value = st.number_input("Seed", value=42, min_value=0) if use_seed else None
        
        if st.button("🚀 Run Bandwidth Sweep", use_container_width=True):
            with st.spinner(f"Running {len(bandwidths)} bandwidth tests × {bw_episodes} episodes..."):
                progress_bar = st.progress(0)
                
                results = run_bandwidth_sweep(
                    bandwidths=bandwidths,
                    n_agents=bw_agents,
                    n_episodes=bw_episodes,
                    n_steps=bw_steps,
                    seed=seed_value
                )
                
                progress_bar.progress(100)
                st.session_state.bandwidth_results = results
                
            st.success("Bandwidth sweep complete!")
            st.rerun()
    
    with col2:
        if st.session_state.bandwidth_results:
            results = st.session_state.bandwidth_results
            
            st.subheader("Results")
            
            fig = create_inverted_u_plot(results)
            st.plotly_chart(fig, use_container_width=True)
            
            has_u_curve, message, peak_idx = detect_inverted_u_curve(results)
            
            if has_u_curve:
                st.success(f"✅ {message}")
                optimal_bw = results[peak_idx]['bandwidth']
                st.info(f"**Optimal Bandwidth:** {optimal_bw} bits achieves peak efficiency of {results[peak_idx]['mean_efficiency']:.2f}")
            else:
                st.warning(f"⚠️ {message}")
            
            st.divider()
            
            df = pd.DataFrame([
                {
                    'Bandwidth': r['bandwidth'],
                    'Mean Efficiency': f"{r['mean_efficiency']:.2f}",
                    'Std Dev': f"{r['std_efficiency']:.2f}",
                    'Coordination Rate': f"{r['mean_coordination']:.1%}",
                    'Msg Delivery': f"{r['mean_msg_delivery']:.1%}"
                }
                for r in results
            ])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            fig_coord = go.Figure()
            fig_coord.add_trace(go.Scatter(
                x=[r['bandwidth'] for r in results],
                y=[r['mean_coordination'] for r in results],
                mode='lines+markers',
                name='Coordination Rate',
                line=dict(color='#e74c3c', width=2)
            ))
            fig_coord.update_layout(
                title='Coordination Rate vs. Bandwidth',
                xaxis_title='Bandwidth (bits)',
                yaxis_title='Coordination Rate',
                xaxis_type='log',
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig_coord, use_container_width=True)
        
        else:
            st.info("Run a bandwidth sweep to see results")

with tab3:
    st.header("Causal Ablation Test (A-B-C Design)")
    st.markdown("**Test causality** by manipulating bandwidth constraints across three phases")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Test Design")
        
        st.markdown("""
        **Phase A:** Constrained bandwidth  
        **Phase B:** Remove constraints (unconstrained)  
        **Phase C:** Re-apply constraints  
        
        If constraints **cause** coordination, we expect:
        - A > B (removing constraints hurts)
        - C > B (restoring constraints helps)
        - A ≈ C (consistency)
        """)
        
        st.divider()
        
        constrained_bw = st.select_slider(
            "Constrained Bandwidth",
            options=[100, 500, 1000, 2000, 5000],
            value=1000
        )
        
        unconstrained_bw = st.select_slider(
            "Unconstrained Bandwidth",
            options=[10000, 50000, 100000, 500000],
            value=100000
        )
        
        causal_agents = st.slider("Agents", 6, 20, 12, key='causal_agents')
        causal_episodes = st.slider("Episodes per phase", 5, 30, 15, key='causal_episodes')
        causal_steps = st.slider("Steps per episode", 30, 60, 40, key='causal_steps')
        causal_seed = st.number_input("Seed", value=42, min_value=0, key='causal_seed')
        
        if st.button("🔬 Run Causal Test", use_container_width=True):
            with st.spinner("Running A-B-C causal ablation test..."):
                progress = st.progress(0)
                
                causal_results = run_causal_ablation_test(
                    bandwidth_constrained=constrained_bw,
                    bandwidth_unconstrained=unconstrained_bw,
                    n_agents=causal_agents,
                    n_episodes=causal_episodes,
                    n_steps=causal_steps,
                    seed=causal_seed
                )
                
                progress.progress(100)
                st.session_state.causal_results = causal_results
                
            st.success("Causal test complete!")
            st.rerun()
    
    with col2:
        if st.session_state.causal_results:
            results = st.session_state.causal_results
            
            st.subheader("Results")
            
            fig = create_phase_comparison_plot(results)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric(
                    "Phase A",
                    f"{results['phase_a']['mean_efficiency']:.2f}",
                    help="Constrained bandwidth"
                )
            
            with col_r2:
                st.metric(
                    "Phase B",
                    f"{results['phase_b']['mean_efficiency']:.2f}",
                    delta=f"{results['drop']:.2f}",
                    delta_color="inverse",
                    help="Unconstrained bandwidth"
                )
            
            with col_r3:
                st.metric(
                    "Phase C",
                    f"{results['phase_c']['mean_efficiency']:.2f}",
                    delta=f"{results['recovery']:.2f}",
                    help="Re-constrained bandwidth"
                )
            
            st.divider()
            st.subheader("Causal Tests")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            
            with col_t1:
                icon = "✅" if results['test_removal_hurts'] else "❌"
                st.metric("Removal Hurts", icon, help="A > B by at least 1.5")
            
            with col_t2:
                icon = "✅" if results['test_restoration_helps'] else "❌"
                st.metric("Restoration Helps", icon, help="C > B by at least 1.5")
            
            with col_t3:
                icon = "✅" if results['test_consistent'] else "❌"
                st.metric("Consistent", icon, help="|A - C| < 2.5")
            
            st.divider()
            
            if results['causality_confirmed']:
                st.success(f"🎆 **CAUSALITY CONFIRMED!** ({results['tests_passed']}/3 tests passed)")
                st.markdown("Bandwidth constraints **cause** coordination emergence")
            else:
                st.warning(f"⚠️ **Causality not established** ({results['tests_passed']}/3 tests passed)")
            
            st.info(f"**Statistical Test:** t-statistic = {results['t_statistic']:.3f}, p-value = {results['p_value']:.4f}")
            
            with st.expander("View Detailed Statistics"):
                st.markdown("**Effect Sizes:**")
                st.write(f"- Constraint removal effect (A→B): {results['drop']:+.2f}")
                st.write(f"- Constraint restoration effect (B→C): {results['recovery']:+.2f}")
                st.write(f"- Consistency (|A-C|): {results['consistency']:.2f}")
        
        else:
            st.info("Run a causal test to see results")

with tab4:
    st.header("Analytics Dashboard")
    
    if st.session_state.current_env and st.session_state.current_env.logger:
        logger = st.session_state.current_env.logger
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Message Flow")
            msg_df = logger.get_message_dataframe()
            
            if not msg_df.empty:
                msg_summary = msg_df.groupby(['step', 'delivered']).size().reset_index(name='count')
                msg_summary['status'] = msg_summary['delivered'].map({True: 'Delivered', False: 'Dropped'})
                
                fig = px.bar(
                    msg_summary,
                    x='step',
                    y='count',
                    color='status',
                    title='Messages per Step',
                    color_discrete_map={'Delivered': '#2ecc71', 'Dropped': '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                msg_type_counts = msg_df[msg_df['delivered']]['message_type'].value_counts()
                fig_pie = px.pie(
                    values=msg_type_counts.values,
                    names=msg_type_counts.index,
                    title='Message Types Delivered'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No message data yet")
        
        with col2:
            st.subheader("Coordination Events")
            coord_df = logger.get_coordination_dataframe()
            
            if not coord_df.empty:
                coord_summary = coord_df.groupby(['step', 'event_type']).size().reset_index(name='count')
                
                fig = px.line(
                    coord_summary,
                    x='step',
                    y='count',
                    color='event_type',
                    title='Events Over Time',
                    color_discrete_map={
                        'food_collected': '#27ae60',
                        'danger_hit': '#c0392b'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                cumulative_food = (coord_df[coord_df['event_type'] == 'food_collected']
                                 .groupby('step').size().cumsum())
                cumulative_danger = (coord_df[coord_df['event_type'] == 'danger_hit']
                                    .groupby('step').size().cumsum())
                
                fig_cum = go.Figure()
                if not cumulative_food.empty:
                    fig_cum.add_trace(go.Scatter(
                        x=cumulative_food.index,
                        y=cumulative_food.values,
                        name='Food (cumulative)',
                        line=dict(color='#27ae60')
                    ))
                if not cumulative_danger.empty:
                    fig_cum.add_trace(go.Scatter(
                        x=cumulative_danger.index,
                        y=cumulative_danger.values,
                        name='Dangers (cumulative)',
                        line=dict(color='#c0392b')
                    ))
                fig_cum.update_layout(title='Cumulative Performance', template='plotly_white')
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.info("No coordination data yet")
        
        st.divider()
        
        move_df = logger.get_movement_dataframe()
        
        if not move_df.empty:
            st.subheader("Movement Analysis")
            
            decision_counts = move_df['decision_basis'].value_counts()
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                fig_decisions = px.pie(
                    values=decision_counts.values,
                    names=decision_counts.index,
                    title='Decision Basis Distribution'
                )
                st.plotly_chart(fig_decisions, use_container_width=True)
            
            with col_m2:
                coord_by_step = move_df.groupby('step')['moved_toward_goal'].mean()
                
                fig_coord_time = go.Figure()
                fig_coord_time.add_trace(go.Scatter(
                    x=coord_by_step.index,
                    y=coord_by_step.values,
                    mode='lines+markers',
                    name='Coordination Rate'
                ))
                fig_coord_time.update_layout(
                    title='Coordination Rate Over Time',
                    yaxis_title='Rate',
                    yaxis_tickformat='.0%',
                    template='plotly_white'
                )
                st.plotly_chart(fig_coord_time, use_container_width=True)
    
    else:
        st.info("Run a simulation to see analytics")

with tab5:
    st.header("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Simulation Logs")
        
        if st.session_state.current_env and st.session_state.current_env.logger:
            logger = st.session_state.current_env.logger
            
            if st.button("Download Message Logs (CSV)"):
                msg_df = logger.get_message_dataframe()
                if not msg_df.empty:
                    csv = msg_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download",
                        csv,
                        "message_logs.csv",
                        "text/csv"
                    )
            
            if st.button("Download Movement Logs (CSV)"):
                move_df = logger.get_movement_dataframe()
                if not move_df.empty:
                    csv = move_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download",
                        csv,
                        "movement_logs.csv",
                        "text/csv"
                    )
            
            if st.button("Download Summary Statistics (JSON)"):
                stats = logger.get_summary_statistics()
                json_str = json.dumps(stats, indent=2)
                st.download_button(
                    "📥 Download",
                    json_str,
                    "summary_stats.json",
                    "application/json"
                )
        else:
            st.info("No simulation data available")
    
    with col2:
        st.subheader("Experiment Results")
        
        if st.session_state.bandwidth_results:
            if st.button("Download Bandwidth Results (CSV)"):
                results = st.session_state.bandwidth_results
                df = pd.DataFrame([
                    {
                        'bandwidth': r['bandwidth'],
                        'mean_efficiency': r['mean_efficiency'],
                        'std_efficiency': r['std_efficiency'],
                        'mean_coordination': r['mean_coordination'],
                        'mean_msg_delivery': r['mean_msg_delivery']
                    }
                    for r in results
                ])
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download",
                    csv,
                    "bandwidth_results.csv",
                    "text/csv"
                )
        
        if st.session_state.causal_results:
            if st.button("Download Causal Test Results (JSON)"):
                results = st.session_state.causal_results
                
                export_data = {
                    'phase_a': {
                        'bandwidth': results['phase_a']['bandwidth'],
                        'mean_efficiency': results['phase_a']['mean_efficiency'],
                        'std_efficiency': results['phase_a']['std_efficiency']
                    },
                    'phase_b': {
                        'bandwidth': results['phase_b']['bandwidth'],
                        'mean_efficiency': results['phase_b']['mean_efficiency'],
                        'std_efficiency': results['phase_b']['std_efficiency']
                    },
                    'phase_c': {
                        'bandwidth': results['phase_c']['bandwidth'],
                        'mean_efficiency': results['phase_c']['mean_efficiency'],
                        'std_efficiency': results['phase_c']['std_efficiency']
                    },
                    'analysis': {
                        'drop': results['drop'],
                        'recovery': results['recovery'],
                        'consistency': results['consistency'],
                        'causality_confirmed': results['causality_confirmed'],
                        'tests_passed': results['tests_passed']
                    }
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    "📥 Download",
                    json_str,
                    "causal_results.json",
                    "application/json"
                )

st.divider()
st.markdown("---")
st.caption("Enhanced Multi-Agent Coordination Simulator | Testing Constraint-Driven Emergence Hypothesis")
