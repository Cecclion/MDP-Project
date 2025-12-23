import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Robot Q-Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional visualization
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Hallway Grid */
    .hallway-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 25px;
        margin: 30px 0;
        padding: 20px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 20px;
    }
    
    /* Cell Styling */
    .cell {
        position: relative;
        aspect-ratio: 1;
        border-radius: 25px;
        padding: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border: 3px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .cell-robot {
        background: linear-gradient(145deg, rgba(139, 92, 246, 0.4), rgba(124, 58, 237, 0.3));
        border: 4px solid #8b5cf6;
        box-shadow: 0 0 40px rgba(139, 92, 246, 0.6), 0 8px 32px rgba(0, 0, 0, 0.2);
        transform: scale(1.08);
    }
    
    .cell-goal {
        background: linear-gradient(145deg, rgba(251, 191, 36, 0.3), rgba(245, 158, 11, 0.2));
        border: 4px solid #fbbf24;
        box-shadow: 0 0 30px rgba(251, 191, 36, 0.5);
    }
    
    .cell-visited {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.25), rgba(37, 99, 235, 0.15));
        border: 3px solid rgba(59, 130, 246, 0.4);
    }
    
    /* State Label */
    .state-label {
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 6px 14px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 13px;
        backdrop-filter: blur(10px);
    }
    
    /* Robot Icon */
    .robot-icon {
        font-size: 90px;
        animation: float 2.5s ease-in-out infinite;
        filter: drop-shadow(0 5px 15px rgba(139, 92, 246, 0.4));
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(5deg); }
    }
    
    /* Lightning Icon */
    .lightning-icon {
        font-size: 85px;
        animation: pulse-glow 2s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(251, 191, 36, 0.8));
    }
    
    @keyframes pulse-glow {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.15); opacity: 0.8; }
    }
    
    /* Action Arrow */
    .action-arrow {
        position: absolute;
        top: 12px;
        right: 12px;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        animation: pulse-arrow 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse-arrow {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    /* Policy Arrow */
    .policy-arrow {
        position: absolute;
        bottom: 15px;
        right: 15px;
        background: rgba(59, 130, 246, 0.9);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0 3px 10px rgba(59, 130, 246, 0.5);
    }
    
    /* Q-Values Display */
    .q-values {
        position: absolute;
        bottom: 12px;
        left: 12px;
        font-size: 10px;
        background: rgba(0, 0, 0, 0.7);
        padding: 4px 8px;
        border-radius: 8px;
        color: #a0aec0;
        font-family: monospace;
        backdrop-filter: blur(10px);
    }
    
    /* Action Badge */
    .action-badge {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        animation: slide-in 0.3s ease-out;
    }
    
    @keyframes slide-in {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Success Badge */
    .success-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 15px 25px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 20px;
        display: inline-block;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(16, 185, 129, 0.5);
        animation: bounce-in 0.5s ease-out;
    }
    
    @keyframes bounce-in {
        0% { transform: scale(0); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)


class QLearningAgent:
    """Q-Learning Agent that learns to navigate the hallway"""
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.states = [0, 1, 2, 3]
        self.actions = ['LEFT', 'RIGHT']
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.terminal_state = 3
        
        # Initialize Q-table with zeros
        self.Q = {state: {'LEFT': 0.0, 'RIGHT': 0.0} for state in self.states}
        
        # Training statistics
        self.episode_count = 0
        self.reward_history = []
        self.steps_history = []
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy"""
        if state == self.terminal_state:
            return None
            
        # Exploration: choose random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: choose best action based on Q-values
        q_left = self.Q[state]['LEFT']
        q_right = self.Q[state]['RIGHT']
        
        if q_left == q_right:
            return random.choice(self.actions)
        
        return 'LEFT' if q_left > q_right else 'RIGHT'
    
    def take_action(self, state, action):
        """Execute action and return next state and reward"""
        if state == self.terminal_state:
            return state, 0
        
        # Stochastic transitions: 80% success, 20% stay in place
        if action == 'RIGHT':
            if random.random() < 0.8:
                next_state = min(state + 1, 3)
            else:
                next_state = state
        else:  # LEFT
            if state == 0:
                next_state = 0
            elif random.random() < 0.8:
                next_state = max(state - 1, 0)
            else:
                next_state = state
        
        # Calculate reward
        if next_state == self.terminal_state:
            reward = 10
        else:
            reward = -1
        
        return next_state, reward
    
    def update_q(self, state, action, reward, next_state):
        """Update Q-value using Q-Learning equation"""
        if state == self.terminal_state:
            return
        
        current_q = self.Q[state][action]
        
        if next_state == self.terminal_state:
            max_next_q = 0
        else:
            max_next_q = max(self.Q[next_state]['LEFT'], self.Q[next_state]['RIGHT'])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state][action] = new_q
    
    def run_episode(self):
        """Run one complete episode of learning"""
        state = random.choice([0, 1, 2])
        episode_reward = 0
        path = [state]
        actions = []
        
        while state != self.terminal_state:
            action = self.choose_action(state)
            next_state, reward = self.take_action(state, action)
            
            self.update_q(state, action, reward, next_state)
            
            episode_reward += reward
            path.append(next_state)
            actions.append(action)
            state = next_state
            
            if len(path) > 100:
                break
        
        self.episode_count += 1
        self.reward_history.append(episode_reward)
        self.steps_history.append(len(path) - 1)
        
        return path, actions, episode_reward
    
    def decay_epsilon(self):
        """Gradually reduce exploration rate"""
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def get_policy(self):
        """Extract the learned policy from Q-table"""
        policy = {}
        for state in self.states:
            if state == self.terminal_state:
                continue
            q_left = self.Q[state]['LEFT']
            q_right = self.Q[state]['RIGHT']
            policy[state] = 'LEFT' if q_left > q_right else 'RIGHT'
        return policy


def visualize_hallway(current_position, policy, visited_states=None, current_action=None, show_q_values=False, agent=None):
    """Enhanced hallway visualization"""
    visited = visited_states if visited_states else []
    
    html = '<div class="hallway-grid">'
    
    for i in range(4):
        # Determine cell class
        cell_classes = ['cell']
        if i == current_position:
            cell_classes.append('cell-robot')
        elif i == 3:
            cell_classes.append('cell-goal')
        elif i in visited and i != current_position:
            cell_classes.append('cell-visited')
        
        html += f'<div class="{" ".join(cell_classes)}">'
        
        # State label
        html += f'<div class="state-label">State {i}</div>'
        
        # Main content
        if i == current_position:
            html += '<div class="robot-icon">🤖</div>'
            if current_action:
                arrow = "←" if current_action == "LEFT" else "→"
                html += f'<div class="action-arrow">{arrow}</div>'
        elif i == 3:
            html += '<div class="lightning-icon">⚡</div>'
        else:
            html += '<div style="font-size: 60px; opacity: 0.2;">▢</div>'
        
        # Policy arrow for non-terminal states
        if i < 3 and i != current_position and i in policy:
            arrow = "←" if policy[i] == "LEFT" else "→"
            html += f'<div class="policy-arrow">{arrow}</div>'
        
        # Q-values display
        if show_q_values and agent and i < 3:
            q_left = agent.Q[i]['LEFT']
            q_right = agent.Q[i]['RIGHT']
            html += f'<div class="q-values">L:{q_left:.1f} R:{q_right:.1f}</div>'
        
        html += '</div>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)


def plot_combined_metrics(agent):
    """Create combined plot with Q-values and performance"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Q-Table Heatmap", "Reward Progress", "Steps per Episode", "Policy Confidence"),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Q-Table Heatmap
    q_data = []
    for state in [0, 1, 2, 3]:
        q_data.append([agent.Q[state]['LEFT'], agent.Q[state]['RIGHT']])
    
    fig.add_trace(
        go.Heatmap(
            z=q_data,
            x=['LEFT ←', 'RIGHT →'],
            y=['State 0', 'State 1', 'State 2', 'State 3'],
            colorscale='Viridis',
            showscale=False,
            text=[[f"{val:.2f}" for val in row] for row in q_data],
            texttemplate="%{text}",
            textfont={"size": 12}
        ),
        row=1, col=1
    )
    
    # Reward Progress
    if len(agent.reward_history) > 0:
        fig.add_trace(
            go.Scatter(
                y=agent.reward_history,
                mode='lines',
                name='Reward',
                line=dict(color='#10b981', width=2)
            ),
            row=1, col=2
        )
        
        # Steps per Episode
        fig.add_trace(
            go.Scatter(
                y=agent.steps_history,
                mode='lines',
                name='Steps',
                line=dict(color='#3b82f6', width=2)
            ),
            row=2, col=1
        )
    
    # Policy Confidence (Q-value differences)
    policy_confidence = []
    states_labels = []
    for state in [0, 1, 2]:
        diff = abs(agent.Q[state]['LEFT'] - agent.Q[state]['RIGHT'])
        policy_confidence.append(diff)
        states_labels.append(f'State {state}')
    
    fig.add_trace(
        go.Bar(
            x=states_labels,
            y=policy_confidence,
            marker_color='#8b5cf6',
            name='Confidence'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig


def show_best_path_demo(agent):
    """Demonstrate the best learned path"""
    st.markdown("### 🎯 Best Path Demonstration")
    
    policy = agent.get_policy()
    
    # Try from each starting position
    for start_pos in [0, 1, 2]:
        st.markdown(f"#### Starting from State {start_pos}")
        
        path = [start_pos]
        actions_taken = []
        current = start_pos
        
        # Follow the policy
        while current != 3 and len(path) < 10:
            action = policy.get(current, 'RIGHT')
            actions_taken.append(action)
            
            # Deterministic move for demonstration
            if action == 'RIGHT':
                current = min(current + 1, 3)
            else:
                current = max(current - 1, 0)
            
            path.append(current)
        
        # Display path
        path_visual = []
        for i, (state, action) in enumerate(zip(path[:-1], actions_taken)):
            arrow = "←" if action == "LEFT" else "→"
            path_visual.append(f"State {state}")
            path_visual.append(f"<span style='color: #10b981; font-weight: bold;'>{arrow} {action}</span>")
        path_visual.append(f"<span style='color: #fbbf24; font-weight: bold;'>State {path[-1]} ⚡ GOAL</span>")
        
        st.markdown(" → ".join(path_visual), unsafe_allow_html=True)
        st.markdown(f"**Total Steps:** {len(path) - 1}")
        st.markdown("---")


# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.3)
    st.session_state.current_path = []
    st.session_state.current_actions = []
    st.session_state.current_position = None
    st.session_state.is_simulating = False
    st.session_state.show_q_values = False


# Header
st.title("🤖 Robot Self-Learning with Q-Learning")
st.markdown("### Watch the robot explore LEFT ← and RIGHT → actions to learn the optimal path!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Learning Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("α (Learning)", f"{st.session_state.agent.alpha:.2f}")
        st.metric("γ (Discount)", f"{st.session_state.agent.gamma:.2f}")
    with col2:
        st.metric("ε (Explore)", f"{st.session_state.agent.epsilon:.3f}")
        st.metric("Episodes", st.session_state.agent.episode_count)
    
    st.divider()
    
    st.header("📊 Performance Metrics")
    
    if len(st.session_state.agent.reward_history) > 0:
        avg_reward = np.mean(st.session_state.agent.reward_history[-20:])
        avg_steps = np.mean(st.session_state.agent.steps_history[-20:])
        best_reward = max(st.session_state.agent.reward_history)
        
        st.metric("Avg Reward (Last 20)", f"{avg_reward:.2f}")
        st.metric("Avg Steps (Last 20)", f"{avg_steps:.1f}")
        st.metric("Best Reward", f"{best_reward:.1f}")
    else:
        st.info("Train the agent to see metrics")
    
    st.divider()
    
    st.header("🎯 MDP Setup")
    st.markdown("""
    **Rewards:**
    - 🎯 Goal: **+10**
    - 👣 Step: **-1**
    
    **Transitions:**
    - ✅ Success: **80%**
    - ⏸️ Stay: **20%**
    
    **Actions:**
    - ← **LEFT**
    - → **RIGHT**
    """)
    
    st.divider()
    
    st.session_state.show_q_values = st.checkbox("Show Q-Values on Grid", value=False)
    
    st.divider()
    
    if st.button("🔄 Reset Agent", use_container_width=True, type="secondary"):
        st.session_state.agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.3)
        st.session_state.current_path = []
        st.session_state.current_actions = []
        st.session_state.current_position = None
        st.rerun()


# Main Control Panel
st.markdown("## ▶️ Control Panel")

control_cols = st.columns(3)

with control_cols[0]:
    if st.button("🔄 Start New Simulation", use_container_width=True, type="primary"):
        start_positions = [0, 1, 2]
        st.session_state.current_position = random.choice(start_positions)
        st.session_state.current_path = [st.session_state.current_position]
        st.session_state.current_actions = []
        st.rerun()

with control_cols[1]:
    if st.button("→ Take Step", use_container_width=True):
        if st.session_state.current_position is not None and st.session_state.current_position != 3:
            agent = st.session_state.agent
            policy = agent.get_policy()
            action = policy.get(st.session_state.current_position, 'RIGHT')
            
            next_state, reward = agent.take_action(st.session_state.current_position, action)
            st.session_state.current_position = next_state
            st.session_state.current_path.append(next_state)
            st.session_state.current_actions.append(action)
            st.rerun()
        elif st.session_state.current_position is None:
            st.warning("⚠️ Please start a new simulation first!")
        else:
            st.success("🎉 Already at goal!")

with control_cols[2]:
    if st.button("⚡ Auto Simulate", use_container_width=True):
        if st.session_state.current_position is not None and st.session_state.current_position != 3:
            agent = st.session_state.agent
            policy = agent.get_policy()
            
            placeholder = st.empty()
            current_pos = st.session_state.current_position
            
            while current_pos != 3:
                action = policy.get(current_pos, 'RIGHT')
                next_state, reward = agent.take_action(current_pos, action)
                
                st.session_state.current_position = next_state
                st.session_state.current_path.append(next_state)
                st.session_state.current_actions.append(action)
                
                with placeholder.container():
                    st.markdown(f'<div class="action-badge">Action: {action} {"←" if action == "LEFT" else "→"}</div>', unsafe_allow_html=True)
                    visualize_hallway(
                        next_state, 
                        policy, 
                        st.session_state.current_path,
                        action,
                        st.session_state.show_q_values,
                        agent
                    )
                
                current_pos = next_state
                time.sleep(0.7)
            
            placeholder.markdown('<div class="success-badge">🎉 GOAL REACHED! 🎉</div>', unsafe_allow_html=True)
            time.sleep(1.5)
            st.rerun()
        elif st.session_state.current_position is None:
            st.warning("⚠️ Please start a new simulation first!")


# Current Status
if st.session_state.current_position is not None:
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("📍 Current Position", f"State {st.session_state.current_position}")
    with stat_cols[1]:
        st.metric("👣 Steps Taken", len(st.session_state.current_path) - 1)
    with stat_cols[2]:
        if st.session_state.current_actions:
            last_action = st.session_state.current_actions[-1]
            arrow = "←" if last_action == "LEFT" else "→"
            st.metric("🎯 Last Action", f"{last_action} {arrow}")
    with stat_cols[3]:
        if st.session_state.current_position == 3:
            st.success("✅ At Goal!")
        else:
            st.info("🏃 Moving...")


# Hallway Visualization
st.markdown("## 🏢 Hallway Environment")

policy = st.session_state.agent.get_policy()
current_pos = st.session_state.current_position
current_action = st.session_state.current_actions[-1] if st.session_state.current_actions else None

visualize_hallway(
    current_pos if current_pos is not None else -1,
    policy,
    st.session_state.current_path,
    current_action,
    st.session_state.show_q_values,
    st.session_state.agent
)

# Path Display
if st.session_state.current_path and len(st.session_state.current_path) > 1:
    st.markdown("### 📍 Current Path Taken:")
    path_elements = []
    for i, (state, action) in enumerate(zip(st.session_state.current_path[:-1], st.session_state.current_actions)):
        arrow = "←" if action == "LEFT" else "→"
        path_elements.append(f"**State {state}**")
        path_elements.append(f"<span style='color: #10b981;'>{arrow} {action}</span>")
    path_elements.append(f"**State {st.session_state.current_path[-1]}**")
    
    st.markdown(" → ".join(path_elements), unsafe_allow_html=True)


# Training Section
st.markdown("---")
st.markdown("## 🧠 Training Section")

train_cols = st.columns(3)

with train_cols[0]:
    if st.button("📚 Train 50 Episodes", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(50):
            st.session_state.agent.run_episode()
            st.session_state.agent.decay_epsilon()
            if (i + 1) % 5 == 0:
                progress_bar.progress((i + 1) / 50)
                status_text.text(f"Training... {i+1}/50 episodes")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Training complete!")
        time.sleep(0.5)
        st.rerun()

with train_cols[1]:
    if st.button("⚡ Train 200 Episodes", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(200):
            st.session_state.agent.run_episode()
            st.session_state.agent.decay_epsilon()
            if (i + 1) % 10 == 0:
                progress_bar.progress((i + 1) / 200)
                status_text.text(f"Training... {i+1}/200 episodes")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Intensive training complete!")
        time.sleep(0.5)
        st.rerun()

with train_cols[2]:
    if st.button("🚀 Train 500 Episodes", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(500):
            st.session_state.agent.run_episode()
            st.session_state.agent.decay_epsilon()
            if (i + 1) % 25 == 0:
                progress_bar.progress((i + 1) / 500)
                status_text.text(f"Training... {i+1}/500 episodes")
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Maximum training complete!")
        time.sleep(0.5)
        st.rerun()


# Best Path After Training
if st.session_state.agent.episode_count > 0:
    st.markdown("---")
    show_best_path_demo(st.session_state.agent)


# Detailed Analytics
st.markdown("---")
st.markdown("## 📊 Detailed Analytics")

if st.session_state.agent.episode_count > 0:
    fig = plot_combined_metrics(st.session_state.agent)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📈 Train the agent to see detailed analytics!")


# Policy Table
st.markdown("---")
st.markdown("## 📋 Learned Policy Table")

policy_data = []
for state in [0, 1, 2]:
    action = policy.get(state, 'N/A')
    arrow = "←" if action == "LEFT" else "→"
    q_left = st.session_state.agent.Q[state]['LEFT']
    q_right = st.session_state.agent.Q[state]['RIGHT']
    best = "✅" if (action == "LEFT" and q_left > q_right) or (action == "RIGHT" and q_right > q_left) else "➖"
    
    policy_data.append({
        'State': f'State {state}',
        'Best Action': f'{action} {arrow}',
                'Q(LEFT)': f'{q_left:.4f}',
        'Q(RIGHT)': f'{q_right:.4f}',
        'Confidence': '✅' if abs(q_left - q_right) > 0.1 else '➖',
        'Optimal': best
    })

policy_df = pd.DataFrame(policy_data)
st.table(policy_df)

# Q-Table Evolution
st.markdown("---")
st.markdown("## 🔄 Q-Table Evolution")

if len(st.session_state.agent.reward_history) > 10:
    # Show Q-value progress over training
    evolution_cols = st.columns(4)
    for state in [0, 1, 2]:
        with evolution_cols[state]:
            st.markdown(f"**State {state} Evolution**")
            
            # Simulate some Q-value progression for demo
            q_left_final = st.session_state.agent.Q[state]['LEFT']
            q_right_final = st.session_state.agent.Q[state]['RIGHT']
            
            progress_left = min(1.0, (q_left_final + 5) / 15)
            progress_right = min(1.0, (q_right_final + 5) / 15)
            
            st.markdown(f"**Q(LEFT):** {q_left_final:.3f}")
            st.progress(progress_left)
            
            st.markdown(f"**Q(RIGHT):** {q_right_final:.3f}")
            st.progress(progress_right)

# Learning Insights
st.markdown("---")
st.markdown("## 🧠 Learning Insights")

insight_cols = st.columns(2)

with insight_cols[0]:
    st.markdown("### 📚 What is Q-Learning?")
    st.markdown("""
    Q-Learning is a model-free **reinforcement learning** algorithm that learns:
    
    1. **Q-values**: Expected future rewards for each (state, action) pair
    2. **Policy**: Which action to take in each state
    3. **Optimal path**: The best sequence of actions to reach the goal
    
    **Key Equation:**
    ```
    Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    ```
    - **α**: Learning rate (how fast to learn)
    - **γ**: Discount factor (value of future rewards)
    - **ε**: Exploration rate (try random actions)
    """)

with insight_cols[1]:
    st.markdown("### 🤔 Learning Observations")
    
    if st.session_state.agent.episode_count == 0:
        st.info("Train the agent to see insights!")
    else:
        # Analyze learning patterns
        policy = st.session_state.agent.get_policy()
        insights = []
        
        # Check if optimal policy is learned
        optimal = True
        for state in [0, 1, 2]:
            if state == 0 and policy.get(state) != 'RIGHT':
                optimal = False
            elif state in [1, 2] and policy.get(state) != 'RIGHT':
                optimal = False
        
        if optimal:
            insights.append("✅ **Optimal policy learned!** Always go RIGHT")
        else:
            insights.append("⚠️ **Sub-optimal policy** - still exploring")
        
        # Check exploration vs exploitation
        if st.session_state.agent.epsilon < 0.05:
            insights.append("🎯 **Exploitation dominant** - mostly using learned policy")
        else:
            insights.append("🔍 **Exploration active** - still trying random actions")
        
        # Check convergence
        if len(st.session_state.agent.reward_history) > 20:
            recent_rewards = st.session_state.agent.reward_history[-20:]
            if np.std(recent_rewards) < 2:
                insights.append("📊 **Converging** - rewards stabilizing")
            else:
                insights.append("📈 **Learning** - rewards still improving")
        
        for insight in insights:
            st.markdown(f"- {insight}")

# Footer
st.markdown("---")
st.markdown("### 🎓 Interactive Learning Simulator")
st.markdown("""
**How to use:**
1. **Start Simulation**: Begin from a random state
2. **Take Step**: Execute one action using current policy
3. **Auto Simulate**: Watch the robot follow learned policy
4. **Train Episodes**: Let the robot learn through experience
5. **Observe**: Watch Q-values update and policy improve

**Learning Goals:**
- Understand how Q-values represent expected rewards
- See exploration vs exploitation trade-off
- Observe policy improvement over time
- Visualize the Markov Decision Process
""")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
