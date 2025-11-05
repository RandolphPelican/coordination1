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
from advanced_visualizations import (
    create_position_heatmap, create_trajectory_visualization, create_decision_heatmap,
    create_agent_efficiency_comparison, create_message_flow_network,
    create_coordination_timeline, create_exploration_coverage_map
)
from statistical_analysis import (
    perform_anova_analysis, perform_regression_analysis,
    perform_comprehensive_causal_analysis, create_regression_plot,
    create_anova_boxplot, create_effect_size_plot, generate_statistical_report
)
from batch_experiments import (
    BatchExperimentRunner, AutomatedReportGenerator, create_batch_template
)
from agent_architectures import AgentArchitectureManager
from rl_agent import RLCoordinationSimulation, QLearningStrategy
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
if 'batch_runner' not in st.session_state:
    st.session_state.batch_runner = BatchExperimentRunner()
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'rl_sim' not in st.session_state:
    st.session_state.rl_sim = None
if 'rl_training_history' not in st.session_state:
    st.session_state.rl_training_history = []

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🎮 Interactive Simulation", 
    "📊 Bandwidth Analysis", 
    "🔬 Causal Testing",
    "📈 Analytics Dashboard",
    "🧠 Behavior Analysis",
    "🔄 Batch Experiments",
    "🤖 Q-Learning",
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
        
        st.subheader("Agent Strategy")
        
        available_strategies = AgentArchitectureManager.get_available_strategies()
        strategy_descriptions = AgentArchitectureManager.get_strategy_descriptions()
        
        agent_strategy = st.selectbox(
            "Strategy Type",
            options=available_strategies,
            format_func=lambda x: x.capitalize(),
            key='agent_strategy_tab1'
        )
        
        st.info(f"**{agent_strategy.capitalize()}:** {strategy_descriptions[agent_strategy]}")
        
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
                    vision_radius=vision_radius,
                    agent_strategy=agent_strategy
                )
                env.initialize(seed=42)
                st.session_state.current_env = env
                st.success(f"Environment initialized with {agent_strategy.capitalize()} agents!")
        
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
            
            st.divider()
            st.subheader("📊 Statistical Analysis")
            
            tab_stats1, tab_stats2 = st.tabs(["ANOVA", "Regression"])
            
            with tab_stats1:
                st.markdown("**One-Way ANOVA:** Testing if bandwidth groups differ significantly")
                
                anova_results = perform_anova_analysis(results)
                
                if anova_results:
                    col_a1, col_a2, col_a3 = st.columns(3)
                    
                    with col_a1:
                        st.metric("F-Statistic", f"{anova_results['f_statistic']:.2f}")
                    with col_a2:
                        st.metric("p-value", f"{anova_results['p_value']:.4f}")
                    with col_a3:
                        sig_icon = "✅" if anova_results['significant'] else "❌"
                        st.metric("Significant (α=0.05)", sig_icon)
                    
                    if anova_results['significant']:
                        st.success("Significant differences detected between bandwidth groups!")
                    else:
                        st.warning("No significant differences between groups")
                    
                    fig_anova = create_anova_boxplot(anova_results)
                    st.plotly_chart(fig_anova, use_container_width=True)
                    
                    with st.expander("View Tukey HSD Post-hoc Test"):
                        st.text(str(anova_results['tukey_hsd']))
                else:
                    st.info("Need at least 2 bandwidth levels for ANOVA")
            
            with tab_stats2:
                st.markdown("**Regression Analysis:** Testing for inverted U-curve relationship")
                
                regression_results = perform_regression_analysis(results)
                
                if regression_results:
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.metric("Linear R²", f"{regression_results['linear_r2']:.3f}")
                    with col_r2:
                        if regression_results['quadratic_r2']:
                            st.metric("Quadratic R²", f"{regression_results['quadratic_r2']:.3f}")
                    
                    if regression_results['has_inverted_u']:
                        st.success(f"✅ Inverted U-curve detected! Optimal bandwidth: {regression_results['optimal_bandwidth']:.0f} bits")
                    else:
                        st.info("No clear inverted U-curve pattern")
                    
                    fig_reg = create_regression_plot(regression_results)
                    st.plotly_chart(fig_reg, use_container_width=True)
                    
                    with st.expander("View OLS Summary"):
                        st.text(str(regression_results['ols_summary']))
                else:
                    st.info("Need at least 3 bandwidth levels for regression")
        
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
            
            st.divider()
            st.subheader("📊 Advanced Statistical Analysis")
            
            causal_stats = perform_comprehensive_causal_analysis(results)
            
            st.markdown("**Pairwise Comparisons (t-tests):**")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            
            with col_t1:
                st.metric("A vs B", 
                         f"p = {causal_stats['t_test_a_vs_b']['p']:.4f}",
                         help=f"t = {causal_stats['t_test_a_vs_b']['t']:.2f}")
            with col_t2:
                st.metric("B vs C",
                         f"p = {causal_stats['t_test_b_vs_c']['p']:.4f}",
                         help=f"t = {causal_stats['t_test_b_vs_c']['t']:.2f}")
            with col_t3:
                st.metric("A vs C",
                         f"p = {causal_stats['t_test_a_vs_c']['p']:.4f}",
                         help=f"t = {causal_stats['t_test_a_vs_c']['t']:.2f}")
            
            st.markdown("**Overall ANOVA:**")
            col_ov1, col_ov2 = st.columns(2)
            
            with col_ov1:
                st.metric("F-Statistic", f"{causal_stats['overall_f']:.2f}")
            with col_ov2:
                st.metric("p-value", f"{causal_stats['overall_p']:.4f}")
            
            st.markdown("**Effect Sizes (Cohen's d):**")
            
            fig_effect = create_effect_size_plot(causal_stats)
            st.plotly_chart(fig_effect, use_container_width=True)
            
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                st.metric("A vs B",
                         f"{causal_stats['effect_size_a_vs_b']['cohens_d']:.2f}",
                         help=causal_stats['effect_size_a_vs_b']['interpretation'])
            with col_e2:
                st.metric("B vs C",
                         f"{causal_stats['effect_size_b_vs_c']['cohens_d']:.2f}",
                         help=causal_stats['effect_size_b_vs_c']['interpretation'])
            with col_e3:
                st.metric("A vs C",
                         f"{causal_stats['effect_size_a_vs_c']['cohens_d']:.2f}",
                         help=causal_stats['effect_size_a_vs_c']['interpretation'])
        
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
    st.header("Behavior Analysis")
    st.markdown("Advanced visualizations for understanding agent behavior patterns and coordination dynamics")
    
    if st.session_state.current_env and st.session_state.current_env.logger:
        logger = st.session_state.current_env.logger
        state = st.session_state.current_env.get_state()
        world_size = state['world_size']
        
        if len(logger.agent_histories) == 0:
            st.info("⏩ Run at least one simulation step to see behavior analysis visualizations")
        else:
            st.subheader("Spatial Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_heatmap = create_position_heatmap(logger, world_size)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No position data yet")
            
            with col2:
                fig_coverage = create_exploration_coverage_map(logger, world_size)
                if fig_coverage:
                    st.plotly_chart(fig_coverage, use_container_width=True)
                else:
                    st.info("No exploration data yet")
            
            st.divider()
            st.subheader("Agent Trajectories")
            
            num_agents_available = len(logger.agent_histories)
            max_agents = st.slider("Number of agents to display", 1, min(num_agents_available, 10), min(5, num_agents_available))
            
            fig_traj = create_trajectory_visualization(logger, world_size, max_agents)
            if fig_traj:
                st.plotly_chart(fig_traj, use_container_width=True)
            else:
                st.info("No trajectory data yet")
        
            st.divider()
            st.subheader("Decision Analysis")
            
            fig_decisions = create_decision_heatmap(logger, world_size)
            if fig_decisions:
                st.plotly_chart(fig_decisions, use_container_width=True)
            else:
                st.info("No decision data yet")
            
            st.divider()
            st.subheader("Performance & Communication")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_efficiency = create_agent_efficiency_comparison(logger)
                if fig_efficiency:
                    st.plotly_chart(fig_efficiency, use_container_width=True)
                else:
                    st.info("No performance data yet")
            
            with col2:
                fig_network = create_message_flow_network(logger)
                if fig_network:
                    st.plotly_chart(fig_network, use_container_width=True)
                else:
                    st.info("No message flow data yet")
            
            st.divider()
            st.subheader("Coordination Timeline")
            
            fig_timeline = create_coordination_timeline(logger)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No timeline data yet")
    
    else:
        st.info("Run a simulation to see behavior analysis")

with tab6:
    st.header("Batch Experiments")
    st.markdown("Run multiple experiments in parallel with automated analysis and reporting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Experiment Configuration")
        
        template_type = st.selectbox(
            "Start from template",
            ["Custom", "Bandwidth Sweep", "Agent Scaling", "Vision Range"]
        )
        
        if template_type != "Custom":
            template_map = {
                "Bandwidth Sweep": "bandwidth_sweep",
                "Agent Scaling": "agent_scaling",
                "Vision Range": "vision_range"
            }
            
            if st.button("Load Template"):
                st.session_state.batch_runner = BatchExperimentRunner()
                template_experiments = create_batch_template(template_map[template_type])
                
                for exp in template_experiments:
                    config = exp['config'].copy()
                    config.setdefault('world_size', 15)
                    config.setdefault('num_agents', 8)
                    config.setdefault('num_food', 10)
                    config.setdefault('num_dangers', 5)
                    config.setdefault('bandwidth_bits', 1000)
                    config.setdefault('vision_radius', 3)
                    config.setdefault('num_steps', 30)
                    config.setdefault('seed', 42)
                    
                    st.session_state.batch_runner.add_experiment(exp['name'], config)
                
                st.success(f"Loaded {len(template_experiments)} experiments from template")
        
        st.divider()
        
        with st.expander("➕ Add Custom Experiment"):
            exp_name = st.text_input("Experiment Name", value=f"Experiment {len(st.session_state.batch_runner.experiments) + 1}")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                num_agents_batch = st.number_input("Agents", 2, 20, 8, key='batch_agents')
                bandwidth_batch = st.number_input("Bandwidth", 100, 100000, 1000, key='batch_bw')
                vision_batch = st.number_input("Vision", 1, 10, 3, key='batch_vision')
            
            with col_c2:
                num_food_batch = st.number_input("Food", 5, 30, 10, key='batch_food')
                num_dangers_batch = st.number_input("Dangers", 2, 15, 5, key='batch_dangers')
                num_runs_batch = st.number_input("Runs per experiment", 1, 20, 5, key='batch_runs')
            
            if st.button("Add to Batch"):
                config = {
                    'world_size': 15,
                    'num_agents': num_agents_batch,
                    'num_food': num_food_batch,
                    'num_dangers': num_dangers_batch,
                    'bandwidth_bits': bandwidth_batch,
                    'vision_radius': vision_batch,
                    'num_steps': 30,
                    'num_runs': num_runs_batch,
                    'seed': 42
                }
                
                st.session_state.batch_runner.add_experiment(exp_name, config)
                st.success(f"Added '{exp_name}' to batch")
                st.rerun()
        
        st.divider()
        
        if st.session_state.batch_runner.experiments:
            st.write(f"**{len(st.session_state.batch_runner.experiments)} experiments** in batch")
            
            if st.button("Clear All"):
                st.session_state.batch_runner = BatchExperimentRunner()
                st.rerun()
        
        st.divider()
        
        if st.session_state.batch_runner.experiments:
            if st.button("🚀 Run Batch", use_container_width=True, type="primary"):
                with st.spinner("Running batch experiments..."):
                    progress_bar = st.progress(0)
                    
                    def update_progress(p):
                        progress_bar.progress(p)
                    
                    results = st.session_state.batch_runner.run_batch(update_progress)
                    st.session_state.batch_results = results
                    
                st.success("Batch complete!")
                st.rerun()
        else:
            st.info("Add experiments to the batch to begin")
    
    with col2:
        if st.session_state.batch_runner.experiments and not st.session_state.batch_results:
            st.subheader("Queued Experiments")
            
            for idx, exp in enumerate(st.session_state.batch_runner.experiments):
                with st.expander(f"📋 {exp['name']}", expanded=(idx == 0)):
                    config = exp['config']
                    
                    col_e1, col_e2, col_e3 = st.columns(3)
                    with col_e1:
                        st.metric("Agents", config.get('num_agents', 8))
                        st.metric("Bandwidth", config.get('bandwidth_bits', 1000))
                    with col_e2:
                        st.metric("Food", config.get('num_food', 10))
                        st.metric("Dangers", config.get('num_dangers', 5))
                    with col_e3:
                        st.metric("Vision", config.get('vision_radius', 3))
                        st.metric("Runs", config.get('num_runs', 5))
        
        elif st.session_state.batch_results:
            st.subheader("Batch Results")
            
            report_gen = AutomatedReportGenerator(st.session_state.batch_results)
            report = report_gen.generate_full_report()
            
            st.markdown("**Executive Summary:**")
            exec_sum = report['executive_summary']
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.success(f"**Best:** {exec_sum['best_experiment']}")
                st.metric("Efficiency", f"{exec_sum['best_efficiency']:.2f}")
            
            with col_r2:
                st.error(f"**Worst:** {exec_sum['worst_experiment']}")
                st.metric("Efficiency", f"{exec_sum['worst_efficiency']:.2f}")
            
            st.metric("Mean Across All", f"{exec_sum['mean_across_all']:.2f} ± {exec_sum['std_across_all']:.2f}")
            
            st.divider()
            
            st.subheader("Comparisons")
            
            tab_comp1, tab_comp2, tab_comp3 = st.tabs(["Efficiency", "Coordination", "Radar"])
            
            with tab_comp1:
                fig_eff = report['visualizations']['efficiency_comparison']
                st.plotly_chart(fig_eff, use_container_width=True)
            
            with tab_comp2:
                fig_coord = report['visualizations']['coordination_comparison']
                st.plotly_chart(fig_coord, use_container_width=True)
            
            with tab_comp3:
                fig_radar = report['visualizations']['radar_chart']
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Radar chart available for ≤5 experiments")
            
            st.divider()
            
            comparison = st.session_state.batch_runner.generate_comparison_report()
            
            if comparison and comparison['statistical_analysis']:
                st.subheader("Statistical Analysis")
                stats = comparison['statistical_analysis']
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    st.metric("F-Statistic", f"{stats['f_statistic']:.2f}")
                with col_s2:
                    st.metric("p-value", f"{stats['p_value']:.4f}")
                with col_s3:
                    sig_icon = "✅" if stats['significant'] else "❌"
                    st.metric("Significant", sig_icon)
            
            st.divider()
            
            st.subheader("Detailed Results")
            summary_df = comparison['summary']
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            if st.button("Clear Results"):
                st.session_state.batch_results = None
                st.session_state.batch_runner = BatchExperimentRunner()
                st.rerun()
        
        else:
            st.info("Configure and run experiments to see results")

with tab7:
    st.header("Q-Learning Adaptive Agents")
    st.markdown("Train agents to learn optimal strategies using reinforcement learning")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Configuration")
        
        rl_num_agents = st.slider("Number of Agents", 2, 8, 4, key='rl_agents')
        rl_num_food = st.slider("Food Items", 5, 20, 10, key='rl_food')
        rl_num_dangers = st.slider("Danger Items", 2, 10, 5, key='rl_dangers')
        rl_vision = st.slider("Vision Radius", 1, 5, 3, key='rl_vision')
        
        st.divider()
        st.subheader("Learning Parameters")
        
        alpha = st.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01, key='rl_alpha')
        gamma = st.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, 0.01, key='rl_gamma')
        epsilon = st.slider("Exploration Rate (ε)", 0.05, 0.5, 0.2, 0.05, key='rl_epsilon')
        
        st.info(f"""
        - **α={alpha}**: How quickly agents learn from new experiences
        - **γ={gamma}**: How much agents value future rewards
        - **ε={epsilon}**: Balance between exploration and exploitation
        """)
        
        st.divider()
        
        if st.button("🎯 Initialize RL Environment", use_container_width=True):
            rl_sim = RLCoordinationSimulation(
                world_size=15,
                num_agents=rl_num_agents,
                num_food=rl_num_food,
                num_dangers=rl_num_dangers,
                vision_radius=rl_vision,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon
            )
            st.session_state.rl_sim = rl_sim
            st.session_state.rl_training_history = []
            st.success("RL environment initialized!")
        
        st.divider()
        
        if st.session_state.rl_sim:
            num_episodes = st.slider("Training Episodes", 1, 100, 10, key='rl_episodes')
            
            if st.button("🚀 Train Agents", use_container_width=True, type="primary"):
                with st.spinner(f"Training for {num_episodes} episodes..."):
                    progress_bar = st.progress(0)
                    
                    for ep in range(num_episodes):
                        stats = st.session_state.rl_sim.run_episode(num_steps=50)
                        st.session_state.rl_training_history.append(stats)
                        progress_bar.progress((ep + 1) / num_episodes)
                
                st.success(f"Training complete! {num_episodes} episodes")
                st.rerun()
            
            if st.button("🔄 Reset Training"):
                st.session_state.rl_training_history = []
                st.session_state.rl_sim = None
                st.rerun()
    
    with col2:
        if st.session_state.rl_training_history:
            st.subheader("Learning Progress")
            
            history = st.session_state.rl_training_history
            episodes = list(range(1, len(history) + 1))
            
            food_collected = [h['total_food'] for h in history]
            dangers_hit = [h['total_dangers'] for h in history]
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Food Collected per Episode', 'Dangers Hit per Episode',
                               'Average Q-Values', 'Total Rewards per Agent')
            )
            
            fig.add_trace(
                go.Scatter(x=episodes, y=food_collected, mode='lines+markers',
                          name='Food', line=dict(color='#2ecc71')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=episodes, y=dangers_hit, mode='lines+markers',
                          name='Dangers', line=dict(color='#e74c3c')),
                row=1, col=2
            )
            
            avg_q_values = []
            for h in history:
                if h['final_q_values']:
                    avg_q_values.append(np.mean(h['final_q_values']))
                else:
                    avg_q_values.append(0)
            
            fig.add_trace(
                go.Scatter(x=episodes, y=avg_q_values, mode='lines+markers',
                          name='Avg Q', line=dict(color='#3498db')),
                row=2, col=1
            )
            
            if history[-1]['agent_rewards']:
                agent_indices = list(range(len(history[-1]['agent_rewards'])))
                fig.add_trace(
                    go.Bar(x=agent_indices, y=history[-1]['agent_rewards'],
                          name='Rewards', marker_color='#9b59b6'),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Episode", row=1, col=1)
            fig.update_xaxes(title_text="Episode", row=1, col=2)
            fig.update_xaxes(title_text="Episode", row=2, col=1)
            fig.update_xaxes(title_text="Agent ID", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("Performance Metrics")
            
            recent_episodes = history[-10:] if len(history) > 10 else history
            
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                avg_food = np.mean([h['total_food'] for h in recent_episodes])
                st.metric("Avg Food (last 10)", f"{avg_food:.1f}")
            
            with col_m2:
                avg_dangers = np.mean([h['total_dangers'] for h in recent_episodes])
                st.metric("Avg Dangers (last 10)", f"{avg_dangers:.1f}")
            
            with col_m3:
                if st.session_state.rl_sim:
                    agent_stats = st.session_state.rl_sim.get_all_agent_stats()
                    avg_epsilon = np.mean([s['epsilon'] for s in agent_stats])
                    st.metric("Current ε", f"{avg_epsilon:.3f}")
            
            st.divider()
            st.subheader("Agent Learning Statistics")
            
            if st.session_state.rl_sim:
                agent_stats = st.session_state.rl_sim.get_all_agent_stats()
                
                stats_data = []
                for i, stats in enumerate(agent_stats):
                    stats_data.append({
                        'Agent': f"Agent {i}",
                        'Total Reward': f"{stats['total_reward']:.1f}",
                        'Avg Q-Value': f"{stats['avg_q_value']:.3f}",
                        'States Explored': stats['states_explored'],
                        'ε': f"{stats['epsilon']:.3f}"
                    })
                
                import pandas as pd
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        elif st.session_state.rl_sim:
            st.info("Initialize and train agents to see learning progress")
        
        else:
            st.info("Initialize RL environment to begin training")

with tab8:
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
