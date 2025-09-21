"""
Reinforcement Learning based scheduler for dynamic train management

This module implements RL approaches for learning optimal scheduling
policies from historical data and real-time feedback.
"""

import logging
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class RLScheduler:
    """
    Reinforcement Learning scheduler using Q-learning approach
    Learns optimal scheduling policies through interaction
    """
    
    def __init__(self):
        """Initialize RL scheduler"""
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.q_table = {}  # State-action value table
        self.action_space_size = 10  # Number of possible actions
        
        logger.info("Reinforcement Learning Scheduler initialized")
    
    def optimize(self, trains, constraints: Dict[str, Any]) -> Dict:
        """
        Optimize using RL-based approach
        
        Args:
            trains: List of Train objects
            constraints: Optimization constraints
            
        Returns:
            Optimization results
        """
        try:
            if not trains:
                return {"status": "success", "schedule": [], "confidence": 1.0}
            
            logger.info(f"Starting RL optimization for {len(trains)} trains")
            
            # Create state representation
            state = self._create_state(trains, constraints)
            
            # Get actions from Q-table or explore
            actions = self._get_actions(state, trains)
            
            # Apply actions to create schedule
            schedule = self._apply_actions(trains, actions)
            
            # Calculate reward (would be updated based on real feedback)
            reward = self._calculate_reward(schedule, trains, constraints)
            
            # Update Q-table (simplified - would use actual feedback in production)
            self._update_q_table(state, actions, reward)
            
            return {
                "status": "success",
                "schedule": schedule,
                "confidence": 0.75,  # RL typically has medium confidence
                "conflicts_resolved": self._count_conflicts_resolved(schedule),
                "delay_reduction": self._calculate_delay_reduction(schedule, trains),
                "reward": reward,
                "exploration_rate": self.epsilon,
                "q_table_size": len(self.q_table)
            }
            
        except Exception as e:
            logger.error(f"RL optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _create_state(self, trains, constraints) -> str:
        """Create state representation for Q-learning"""
        # Simplified state representation
        state_features = []
        
        # Number of trains
        state_features.append(f"trains_{min(len(trains), 10)}")
        
        # Average delay
        avg_delay = sum(train.current_delay for train in trains) / len(trains) if trains else 0
        delay_bucket = min(int(avg_delay // 5), 6)  # 0-30 minutes in 5-minute buckets
        state_features.append(f"delay_{delay_bucket}")
        
        # Priority distribution
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for train in trains:
            priority_counts[train.priority] += 1
        
        high_priority_count = priority_counts[1] + priority_counts[2]
        state_features.append(f"high_priority_{min(high_priority_count, 5)}")
        
        # Time of day (simplified)
        current_hour = datetime.now().hour
        time_period = "morning" if 6 <= current_hour < 12 else \
                     "afternoon" if 12 <= current_hour < 18 else \
                     "evening" if 18 <= current_hour < 22 else "night"
        state_features.append(f"time_{time_period}")
        
        return "_".join(state_features)
    
    def _get_actions(self, state: str, trains) -> List[int]:
        """Get actions using epsilon-greedy policy"""
        actions = []
        
        for i, train in enumerate(trains):
            if random.random() < self.epsilon:
                # Explore: random action
                action = random.randint(0, self.action_space_size - 1)
            else:
                # Exploit: best known action
                state_action_key = f"{state}_{i}"
                if state_action_key in self.q_table:
                    action = max(self.q_table[state_action_key], 
                               key=self.q_table[state_action_key].get)
                else:
                    action = random.randint(0, self.action_space_size - 1)
            
            actions.append(action)
        
        return actions
    
    def _apply_actions(self, trains, actions) -> List[Dict]:
        """Apply RL actions to create schedule"""
        schedule = []
        
        action_mappings = {
            0: {"delay_adjust": 0, "platform": 1, "speed_boost": False},      # No change, platform 1
            1: {"delay_adjust": -5, "platform": 1, "speed_boost": True},     # Speed up, platform 1
            2: {"delay_adjust": 5, "platform": 1, "speed_boost": False},     # Slow down, platform 1
            3: {"delay_adjust": 0, "platform": 2, "speed_boost": False},     # No change, platform 2
            4: {"delay_adjust": -5, "platform": 2, "speed_boost": True},     # Speed up, platform 2
            5: {"delay_adjust": 5, "platform": 2, "speed_boost": False},     # Slow down, platform 2
            6: {"delay_adjust": 0, "platform": 3, "speed_boost": False},     # No change, platform 3
            7: {"delay_adjust": -10, "platform": 1, "speed_boost": True},    # High speed boost, platform 1
            8: {"delay_adjust": 10, "platform": 1, "speed_boost": False},    # High delay, platform 1
            9: {"delay_adjust": 0, "platform": 4, "speed_boost": False},     # No change, platform 4
        }
        
        for i, train in enumerate(trains):
            action = actions[i]
            action_params = action_mappings.get(action, action_mappings[0])
            
            # Apply action parameters
            delay_adjust = action_params["delay_adjust"]
            platform = action_params["platform"]
            speed_boost = action_params["speed_boost"]
            
            # Calculate new timing
            optimized_departure = train.departure_time + timedelta(minutes=delay_adjust)
            
            # Speed boost reduces journey time
            journey_time = train.arrival_time - train.departure_time
            if speed_boost:
                journey_time = journey_time * 0.9  # 10% faster
            
            optimized_arrival = optimized_departure + journey_time
            
            schedule.append({
                "train_id": train.train_id,
                "train_number": train.train_number,
                "original_departure": train.departure_time.isoformat(),
                "optimized_departure": optimized_departure.isoformat(),
                "original_arrival": train.arrival_time.isoformat(),
                "optimized_arrival": optimized_arrival.isoformat(),
                "assigned_platform": platform,
                "delay_minutes": max(0, train.current_delay + delay_adjust),
                "priority": train.priority,
                "action_taken": action,
                "speed_boost": speed_boost
            })
        
        return schedule
    
    def _calculate_reward(self, schedule: List[Dict], trains, constraints) -> float:
        """Calculate reward for the current schedule"""
        reward = 0.0
        
        # Reward for delay reduction
        total_original_delay = sum(train.current_delay for train in trains)
        total_new_delay = sum(s['delay_minutes'] for s in schedule)
        delay_improvement = total_original_delay - total_new_delay
        reward += delay_improvement * 2  # 2 points per minute of delay reduction
        
        # Penalty for conflicts
        conflicts = self._count_platform_conflicts(schedule)
        reward -= conflicts * 10  # -10 points per conflict
        
        # Reward for priority handling
        for s in schedule:
            train = next(t for t in trains if t.train_id == s['train_id'])
            if train.priority <= 2 and s['delay_minutes'] <= 5:  # High priority, low delay
                reward += 5
        
        # Penalty for excessive delays
        for s in schedule:
            if s['delay_minutes'] > 30:
                reward -= 15  # Heavy penalty for long delays
        
        # Reward for balanced platform usage
        platform_counts = {}
        for s in schedule:
            platform = s['assigned_platform']
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        if platform_counts:
            avg_usage = len(schedule) / len(platform_counts)
            balance_score = sum(abs(count - avg_usage) for count in platform_counts.values())
            reward -= balance_score  # Penalty for imbalanced usage
        
        return reward
    
    def _update_q_table(self, state: str, actions: List[int], reward: float):
        """Update Q-table with observed reward"""
        for i, action in enumerate(actions):
            state_action_key = f"{state}_{i}"
            
            if state_action_key not in self.q_table:
                self.q_table[state_action_key] = {a: 0.0 for a in range(self.action_space_size)}
            
            # Q-learning update rule
            current_q = self.q_table[state_action_key][action]
            # Simplified: assume max future Q is 0 (would compute actual next state in practice)
            max_future_q = 0.0
            
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
            self.q_table[state_action_key][action] = new_q
    
    def _count_conflicts_resolved(self, schedule: List[Dict]) -> int:
        """Count conflicts that were resolved"""
        # Simplified: count number of trains that had their delays reduced
        return len([s for s in schedule if s['delay_minutes'] < 10])
    
    def _calculate_delay_reduction(self, schedule: List[Dict], trains) -> int:
        """Calculate total delay reduction"""
        original_delays = sum(train.current_delay for train in trains)
        new_delays = sum(s['delay_minutes'] for s in schedule)
        return max(0, original_delays - new_delays)
    
    def _count_platform_conflicts(self, schedule: List[Dict]) -> int:
        """Count platform conflicts in the schedule"""
        conflicts = 0
        
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                s1, s2 = schedule[i], schedule[j]
                
                if s1['assigned_platform'] == s2['assigned_platform']:
                    # Check time overlap
                    dep1 = datetime.fromisoformat(s1['optimized_departure'].replace('Z', '+00:00'))
                    arr1 = datetime.fromisoformat(s1['optimized_arrival'].replace('Z', '+00:00'))
                    dep2 = datetime.fromisoformat(s2['optimized_departure'].replace('Z', '+00:00'))
                    arr2 = datetime.fromisoformat(s2['optimized_arrival'].replace('Z', '+00:00'))
                    
                    # Check if times overlap
                    if not (arr1 < dep2 or arr2 < dep1):
                        conflicts += 1
        
        return conflicts
    
    def update_from_feedback(self, state: str, actions: List[int], actual_reward: float):
        """Update Q-table from real-world feedback"""
        self._update_q_table(state, actions, actual_reward)
        logger.info(f"Updated Q-table from feedback: reward = {actual_reward}")
    
    def get_policy_summary(self) -> Dict:
        """Get summary of learned policy"""
        return {
            "q_table_size": len(self.q_table),
            "exploration_rate": self.epsilon,
            "learning_rate": self.learning_rate,
            "most_common_actions": self._get_action_distribution(),
            "average_q_values": self._get_average_q_values()
        }
    
    def _get_action_distribution(self) -> Dict:
        """Get distribution of actions in Q-table"""
        action_counts = {i: 0 for i in range(self.action_space_size)}
        
        for state_actions in self.q_table.values():
            best_action = max(state_actions, key=state_actions.get)
            action_counts[best_action] += 1
        
        return action_counts
    
    def _get_average_q_values(self) -> Dict:
        """Get average Q-values for each action"""
        action_values = {i: [] for i in range(self.action_space_size)}
        
        for state_actions in self.q_table.values():
            for action, value in state_actions.items():
                action_values[action].append(value)
        
        return {action: sum(values) / len(values) if values else 0 
                for action, values in action_values.items()}