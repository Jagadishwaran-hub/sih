"""
OR-Tools based optimization scheduler for railway operations

This module uses Google OR-Tools constraint programming and optimization
to solve train scheduling and conflict resolution problems.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

logger = logging.getLogger(__name__)

class ORToolsScheduler:
    """
    OR-Tools based optimization scheduler for train operations
    Uses constraint programming and linear programming for optimization
    """
    
    def __init__(self):
        """Initialize OR-Tools scheduler"""
        self.solver_timeout_seconds = 30
        self.max_conflicts = 100
        logger.info("OR-Tools Scheduler initialized")
    
    def optimize(self, trains, constraints: Dict[str, Any]) -> Dict:
        """
        Optimize train schedule using OR-Tools CP-SAT solver
        
        Args:
            trains: List of Train objects
            constraints: Optimization constraints
            
        Returns:
            Optimization results with schedule
        """
        try:
            if not trains:
                return {"status": "success", "schedule": [], "confidence": 1.0}
            
            logger.info(f"Starting OR-Tools optimization for {len(trains)} trains")
            
            # Create CP-SAT model
            model = cp_model.CpModel()
            
            # Variables and constraints
            schedule_vars = self._create_schedule_variables(model, trains, constraints)
            self._add_capacity_constraints(model, schedule_vars, trains, constraints)
            self._add_precedence_constraints(model, schedule_vars, trains, constraints)
            
            # Objective: minimize total delay and maximize priority satisfaction
            objective = self._create_objective(model, schedule_vars, trains, constraints)
            model.Minimize(objective)
            
            # Solve the model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = self.solver_timeout_seconds
            
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Extract solution
                schedule = self._extract_schedule_solution(solver, schedule_vars, trains)
                confidence = 0.95 if status == cp_model.OPTIMAL else 0.85
                
                return {
                    "status": "success",
                    "schedule": schedule,
                    "confidence": confidence,
                    "conflicts_resolved": self._count_conflicts_resolved(schedule),
                    "delay_reduction": self._calculate_delay_reduction(schedule, trains),
                    "solver_status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                    "solve_time": solver.WallTime()
                }
            else:
                logger.warning(f"OR-Tools solver failed with status: {status}")
                return self._create_fallback_schedule(trains)
                
        except Exception as e:
            logger.error(f"OR-Tools optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def resolve_conflict(self, train1, train2, location: str, conflict_type: str) -> Dict:
        """
        Resolve conflict between two trains using OR-Tools
        
        Args:
            train1: First train object
            train2: Second train object
            location: Conflict location
            conflict_type: Type of conflict
            
        Returns:
            Conflict resolution decision
        """
        try:
            logger.info(f"Resolving {conflict_type} conflict between {train1.train_number} and {train2.train_number}")
            
            # Simple precedence decision based on multiple factors
            score1 = self._calculate_precedence_score(train1)
            score2 = self._calculate_precedence_score(train2)
            
            if score1 > score2:
                winner = train1
                reason = f"Train {train1.train_number} has higher precedence score ({score1:.2f} vs {score2:.2f})"
            elif score2 > score1:
                winner = train2
                reason = f"Train {train2.train_number} has higher precedence score ({score2:.2f} vs {score1:.2f})"
            else:
                # Tie-breaker: train with higher delay gets priority for recovery
                if train1.current_delay > train2.current_delay:
                    winner = train1
                    reason = f"Train {train1.train_number} needs delay recovery"
                else:
                    winner = train2
                    reason = f"Train {train2.train_number} needs delay recovery"
            
            # Calculate time adjustment
            time_adjustment = self._calculate_time_adjustment(winner, conflict_type)
            
            return {
                "status": "resolved",
                "precedence_train_id": winner.train_id,
                "precedence_train_number": winner.train_number,
                "delayed_train_id": train2.train_id if winner == train1 else train1.train_id,
                "reasoning": reason,
                "confidence": 0.9,
                "time_adjustment_minutes": time_adjustment,
                "method": "OR_TOOLS_PRECEDENCE_SCORING"
            }
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    # Helper methods for OR-Tools implementation
    
    def _create_schedule_variables(self, model, trains, constraints) -> Dict:
        """Create scheduling variables for the CP model"""
        schedule_vars = {}
        
        # Time horizon (24 hours in minutes)
        time_horizon = 24 * 60
        
        for train in trains:
            train_id = train.train_id
            
            # Departure time variable
            departure_var = model.NewIntVar(0, time_horizon, f'departure_{train_id}')
            
            # Arrival time variable
            arrival_var = model.NewIntVar(0, time_horizon, f'arrival_{train_id}')
            
            # Platform assignment variable (assuming 4 platforms)
            platform_var = model.NewIntVar(1, 4, f'platform_{train_id}')
            
            # Delay variable
            delay_var = model.NewIntVar(0, 120, f'delay_{train_id}')  # Max 2 hours delay
            
            schedule_vars[train_id] = {
                'departure': departure_var,
                'arrival': arrival_var,
                'platform': platform_var,
                'delay': delay_var
            }
            
            # Journey time constraint
            journey_minutes = int((train.arrival_time - train.departure_time).total_seconds() / 60)
            model.Add(arrival_var == departure_var + journey_minutes)
        
        return schedule_vars
    
    def _add_capacity_constraints(self, model, schedule_vars, trains, constraints):
        """Add platform and track capacity constraints"""
        # Platform capacity constraints
        for platform in range(1, 5):  # 4 platforms
            for time_slot in range(0, 24 * 60, 30):  # 30-minute slots
                platform_users = []
                
                for train in trains:
                    train_id = train.train_id
                    vars = schedule_vars[train_id]
                    
                    # Check if train uses this platform at this time
                    platform_match = model.NewBoolVar(f'platform_match_{train_id}_{platform}_{time_slot}')
                    model.Add(vars['platform'] == platform).OnlyEnforceIf(platform_match)
                    model.Add(vars['platform'] != platform).OnlyEnforceIf(platform_match.Not())
                    
                    time_match = model.NewBoolVar(f'time_match_{train_id}_{platform}_{time_slot}')
                    model.Add(vars['departure'] <= time_slot).OnlyEnforceIf(time_match)
                    model.Add(vars['arrival'] >= time_slot).OnlyEnforceIf(time_match)
                    model.Add(vars['departure'] > time_slot).OnlyEnforceIf(time_match.Not())
                    
                    both_match = model.NewBoolVar(f'both_match_{train_id}_{platform}_{time_slot}')
                    model.AddBoolAnd([platform_match, time_match]).OnlyEnforceIf(both_match)
                    
                    platform_users.append(both_match)
                
                # At most 1 train per platform at any time
                model.Add(sum(platform_users) <= 1)
    
    def _add_precedence_constraints(self, model, schedule_vars, trains, constraints):
        """Add precedence constraints based on train priorities"""
        priority_groups = {}
        
        # Group trains by priority
        for train in trains:
            priority = train.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(train)
        
        # Higher priority trains (lower number) should have preference
        for priority1 in priority_groups:
            for priority2 in priority_groups:
                if priority1 < priority2:  # priority1 is higher
                    for train1 in priority_groups[priority1]:
                        for train2 in priority_groups[priority2]:
                            # If trains conflict, higher priority goes first
                            self._add_conflict_constraint(model, schedule_vars, train1, train2)
    
    def _add_conflict_constraint(self, model, schedule_vars, train1, train2):
        """Add constraint to prevent conflicting trains from overlapping"""
        vars1 = schedule_vars[train1.train_id]
        vars2 = schedule_vars[train2.train_id]
        
        # If same platform, ensure no time overlap
        same_platform = model.NewBoolVar(f'same_platform_{train1.train_id}_{train2.train_id}')
        model.Add(vars1['platform'] == vars2['platform']).OnlyEnforceIf(same_platform)
        
        # If same platform, one must finish before the other starts
        train1_first = model.NewBoolVar(f'train1_first_{train1.train_id}_{train2.train_id}')
        model.Add(vars1['arrival'] + 5 <= vars2['departure']).OnlyEnforceIf([same_platform, train1_first])
        model.Add(vars2['arrival'] + 5 <= vars1['departure']).OnlyEnforceIf([same_platform, train1_first.Not()])
    
    def _create_objective(self, model, schedule_vars, trains, constraints):
        """Create optimization objective"""
        objective_terms = []
        
        for train in trains:
            train_id = train.train_id
            vars = schedule_vars[train_id]
            
            # Minimize delay
            delay_weight = constraints.get('priority_weights', {}).get(train.priority, 1)
            objective_terms.append(vars['delay'] * delay_weight)
            
            # Penalize late departures
            original_departure_minutes = train.departure_time.hour * 60 + train.departure_time.minute
            late_departure = model.NewIntVar(0, 24 * 60, f'late_departure_{train_id}')
            model.AddMaxEquality(late_departure, [vars['departure'] - original_departure_minutes, 0])
            objective_terms.append(late_departure * delay_weight)
        
        return sum(objective_terms)
    
    def _extract_schedule_solution(self, solver, schedule_vars, trains) -> List[Dict]:
        """Extract optimized schedule from solver solution"""
        schedule = []
        
        for train in trains:
            train_id = train.train_id
            vars = schedule_vars[train_id]
            
            departure_minutes = solver.Value(vars['departure'])
            arrival_minutes = solver.Value(vars['arrival'])
            platform = solver.Value(vars['platform'])
            delay = solver.Value(vars['delay'])
            
            # Convert minutes back to datetime
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            optimized_departure = base_date + timedelta(minutes=departure_minutes)
            optimized_arrival = base_date + timedelta(minutes=arrival_minutes)
            
            schedule.append({
                "train_id": train_id,
                "train_number": train.train_number,
                "original_departure": train.departure_time.isoformat(),
                "optimized_departure": optimized_departure.isoformat(),
                "original_arrival": train.arrival_time.isoformat(),
                "optimized_arrival": optimized_arrival.isoformat(),
                "assigned_platform": platform,
                "delay_minutes": delay,
                "priority": train.priority
            })
        
        return schedule
    
    def _calculate_precedence_score(self, train) -> float:
        """Calculate precedence score for conflict resolution"""
        score = 0.0
        
        # Priority factor (higher priority = higher score)
        score += (6 - train.priority) * 20  # Priority 1 gets 100 points, priority 5 gets 20
        
        # Delay factor (more delayed trains get higher score for recovery)
        score += min(train.current_delay * 2, 60)  # Max 60 points for delay
        
        # Train type factor
        type_scores = {"EXPRESS": 30, "PASSENGER": 20, "FREIGHT": 10}
        score += type_scores.get(train.train_type, 15)
        
        # Journey time factor (longer journeys get slight priority)
        journey_hours = (train.arrival_time - train.departure_time).total_seconds() / 3600
        score += min(journey_hours * 5, 25)  # Max 25 points
        
        return score
    
    def _calculate_time_adjustment(self, winning_train, conflict_type: str) -> int:
        """Calculate time adjustment needed for conflict resolution"""
        if conflict_type == "PRECEDENCE":
            return 5  # 5 minute buffer
        elif conflict_type == "CROSSING":
            return 10  # 10 minute buffer for crossing
        elif conflict_type == "PLATFORM":
            return 3  # 3 minute buffer for platform
        else:
            return 5  # Default buffer
    
    def _count_conflicts_resolved(self, schedule: List[Dict]) -> int:
        """Count number of conflicts resolved in the schedule"""
        # Simplified calculation - would be more complex in real implementation
        return len([s for s in schedule if s['delay_minutes'] > 0])
    
    def _calculate_delay_reduction(self, schedule: List[Dict], original_trains) -> int:
        """Calculate total delay reduction achieved"""
        original_delay = sum(train.current_delay for train in original_trains)
        optimized_delay = sum(s['delay_minutes'] for s in schedule)
        return max(0, original_delay - optimized_delay)
    
    def _create_fallback_schedule(self, trains) -> Dict:
        """Create fallback schedule when optimization fails"""
        schedule = []
        
        for train in trains:
            schedule.append({
                "train_id": train.train_id,
                "train_number": train.train_number,
                "original_departure": train.departure_time.isoformat(),
                "optimized_departure": train.departure_time.isoformat(),
                "original_arrival": train.arrival_time.isoformat(),
                "optimized_arrival": train.arrival_time.isoformat(),
                "assigned_platform": 1,  # Default platform
                "delay_minutes": train.current_delay,
                "priority": train.priority
            })
        
        return {
            "status": "fallback",
            "schedule": schedule,
            "confidence": 0.5,
            "conflicts_resolved": 0,
            "delay_reduction": 0,
            "solver_status": "FALLBACK"
        }