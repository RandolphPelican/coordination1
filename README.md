# Multi-Agent Coordination Simulator

A research-grade simulation framework for investigating constraint-driven emergence in multi-agent communication networks.

## Overview

This tool simulates autonomous agents navigating a 2D grid world where they must collect food and avoid dangers while communicating through bandwidth-constrained message channels. The core research question: How do communication bandwidth constraints affect emergent coordination behaviors?

The system tests for inverted U-curve relationships between bandwidth and coordination efficiency - the hypothesis that moderate constraints lead to optimal coordination.

## Features

- Interactive web interface powered by Streamlit
- 6 distinct agent strategies (Greedy, Cautious, Balanced, Explorer, Cooperative, Q-Learning)
- Built-in statistical analysis (ANOVA, regression analysis, effect size calculations)
- Batch experimentation system with automated report generation
- Advanced visualizations (position heatmaps, agent trajectories, decision analysis, message flow networks)
- Causal testing framework (A-B-C intervention analysis)
- Reinforcement learning implementation (Q-learning with configurable parameters)
- Data export functionality (CSV and JSON formats)

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

Clone the repository:
```

git clone <https://github.com/RandolphPelican/coordination1.git>
cd coordination1

```
Install dependencies:
```

pip install -r requirements.txt

```
Or install manually:
```

pip install streamlit numpy pandas scipy statsmodels plotly

```
## Usage

### Running the Simulator

Start the web application:
```

streamlit run app.py

```
The interface will open in your browser at http://localhost:8501

### Quick Start Guide

**Tab 1: Interactive Simulation**
- Configure agent parameters (strategy, count, vision radius)
- Set environment parameters (world size, food, dangers)
- Adjust bandwidth constraints
- Run real-time simulations with visualization

**Tab 2: Bandwidth Analysis**
- Test the inverted U-curve hypothesis
- Run automated bandwidth sweeps
- View statistical analysis (ANOVA, regression)
- Export results for publication

**Tab 3: Causal Testing**
- Run A-B-C intervention experiments
- Test whether constraints CAUSE coordination
- Calculate effect sizes and statistical significance

**Tab 4: Analytics Dashboard**
- View performance metrics over time
- Track message delivery rates
- Analyze coordination patterns
- Compare agent efficiency

**Tab 5: Behavior Analysis**
- Position heatmaps (where agents spend time)
- Trajectory visualization (movement patterns)
- Decision heatmaps (spatial decision analysis)
- Message flow networks (communication patterns)

**Tab 6: Batch Experiments**
- Configure multi-parameter sweeps
- Run large-scale experiments automatically
- Generate formatted research reports
- Export comprehensive datasets

**Tab 7: Q-Learning**
- Train reinforcement learning agents
- Visualize Q-value evolution
- Test adaptive strategies under bandwidth constraints
- Compare learned vs hand-coded strategies

**Tab 8: Data Export**
- Export raw simulation data (CSV)
- Export experiment configurations (JSON)
- Download agent trajectories and message logs
- Generate reproducibility packages

## Example Experiments

### Testing the Inverted U-Curve

1. Go to Tab 2 (Bandwidth Analysis)
2. Select "Detailed (7 points)" for bandwidth range
3. Set Agent Strategy to "Cooperative"
4. Set Agents to 12, Episodes to 20
5. Click "Run Bandwidth Sweep"
6. Examine the graph for peak at moderate bandwidth

### Comparing Agent Strategies

1. Go to Tab 1 (Interactive Simulation)
2. Run simulation with Strategy = "Greedy"
3. Record efficiency score
4. Change Strategy to "Cautious", "Balanced", "Cooperative"
5. Compare performance under same bandwidth

### Causal Intervention Test

1. Go to Tab 3 (Causal Testing)
2. Set Phase A Bandwidth = 1000 bits
3. Set Phase B Bandwidth = 100000 bits (unlimited)
4. Set Phase C Bandwidth = 1000 bits
5. Run test - if A > B and C > B, constraint is causal

## Research Applications

This framework is designed for investigating:

- Emergence in multi-agent systems
- Communication network optimization
- Swarm robotics coordination
- Distributed artificial intelligence
- Collective intelligence mechanisms
- Information theory in agent coordination
- Bandwidth constraints in real-world systems

## Technical Architecture

**Core Components:**

- simulation_core.py - Main simulation engine with event-driven loop
- agent_architectures.py - Pluggable agent strategy implementations
- statistical_analysis.py - ANOVA, regression, and causal analysis tools
- advanced_visualizations.py - Heatmaps, trajectories, network graphs
- batch_experiments.py - Automated parameter sweep system
- rl_agent.py - Q-learning reinforcement learning implementation
- simulation_logger.py - Comprehensive event logging system

**Design Principles:**

- Modular architecture with strategy pattern for agents
- Centralized state management for reproducibility
- Event-driven simulation with deterministic random seeding
- Priority-based message routing with bandwidth budgeting
- In-memory data storage for fast iteration

## Configuration Parameters

**Environment:**
- world_size: Grid dimensions (default: 15x15)
- food_count: Number of food items (default: 12)
- danger_count: Number of dangers (default: 8)

**Agents:**
- agent_count: Number of agents (default: 12)
- vision_radius: How far agents can see (default: 3)
- strategy: Decision-making approach (6 options)

**Communication:**
- bandwidth_bits: Bits available per timestep (default: 1000)
- message_size: Bits per message (default: 100)
- self_censorship: Agents adapt to delivery failure

**Simulation:**
- steps_per_episode: Timesteps in each run (default: 50)
- episodes: Number of replications (default: 10)
- random_seed: For reproducibility (optional)

## Data Export Format

**CSV Output Includes:**
- Episode number and timestep
- Agent positions and states
- Messages sent and delivered
- Food collected and dangers encountered
- Coordination events and efficiency metrics

**JSON Output Includes:**
- Complete parameter configuration
- Agent strategy specifications
- Statistical test results
- Experiment metadata

## Performance Considerations

**Typical Performance:**
- Single episode (12 agents, 50 steps): ~0.5 seconds
- Bandwidth sweep (7 points, 20 episodes each): ~2 minutes
- Batch experiment (100 configurations): ~15 minutes
- Q-learning training (1000 episodes): ~5 minutes

**Optimization Tips:**
- Use fixed random seeds for reproducibility
- Reduce episodes for initial testing
- Use batch experiments for large parameter sweeps
- Export data for offline analysis

## Troubleshooting

**Issue: Simulation is slow**
- Reduce number of agents
- Decrease steps per episode
- Lower batch experiment size

**Issue: No inverted U-curve detected**
- Increase agent count (more messages = bandwidth matters more)
- Use "Cooperative" strategy (most communication-dependent)
- Adjust food/danger ratio (avoid survival mode)
- Add message noise at high bandwidth

**Issue: All bandwidths perform similarly**
- Check if agents are using messages (view message delivery rate)
- Reduce vision radius (increase dependence on communication)
- Ensure bandwidth range is sufficiently wide (50 to 50000 bits)

## Citation

If you use this tool in your research, please cite:
```

@software{coordination_simulator_2025,
John Stabler
Multi-Agent Coordination Simulator: A Framework for Testing Constraint-Driven Emergence
2025
https://github.com/RandolphPelican/coordination1
version = {1.0}
}

```
## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

Questions, bug reports, or research collaborations:
- Open an issue on GitHub
- Email: Johndstabler@gmail.com
- Twitter:@SkilletBenz

## Acknowledgments

Built with:
- Streamlit (web framework)
- NumPy & Pandas (data processing)
- SciPy & statsmodels (statistical analysis)
- Plotly (interactive visualizations)

Inspired by research in:
- Complexity science
- Multi-agent systems
- Emergence theory
- Information theory
- Swarm intelligence
_ Scale Gap analysis 
## Roadmap

**Planned Features:**
- 3D environment visualization
- Additional agent learning algorithms (DQN, PPO)
- Real-time collaboration mode (multiple users)
- Custom agent strategy builder (GUI)
- Integration with physics engines
- Network topology variations
- Mobile app version

## Version History

**v1.0.0 (November 2025)**
- Initial release
- 6 agent strategies
- 8 interactive tabs
- Comprehensive statistical testing
- Batch experimentation system
- Q-learning implementation
- Advanced visualizations
