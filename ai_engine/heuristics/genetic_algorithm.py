"""
Genetic Algorithm based optimization for train scheduling

This module implements genetic algorithm heuristics for optimizing
train schedules when exact solutions are not feasible.
"""

import logging
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class GeneticAlgorithmScheduler:
    """
    Genetic Algorithm implementation for train scheduling optimization
    Uses evolutionary algorithms to find near-optimal solutions
    """
    
    def __init__(self):
        """Initialize Genetic Algorithm scheduler"""
        self.population_size = 50
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elite_size = 5
        
        logger.info("Genetic Algorithm Scheduler initialized")
    
    def optimize(self, trains, constraints: Dict[str, Any]) -> Dict:
        """
        Optimize train schedule using Genetic Algorithm
        
        Args:
            trains: List of Train objects
            constraints: Optimization constraints
            
        Returns:
            Optimization results with schedule
        """
        try:
            if not trains:
                return {"status": "success", "schedule": [], "confidence": 1.0}
            
            logger.info(f"Starting GA optimization for {len(trains)} trains")
            
            # Initialize population
            population = self._initialize_population(trains, constraints)
            
            best_fitness_history = []
            best_individual = None
            
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = [self._evaluate_fitness(individual, trains, constraints) 
                                for individual in population]
                
                # Track best individual
                best_idx = np.argmax(fitness_scores)
                if best_individual is None or fitness_scores[best_idx] > max(best_fitness_history or [0]):
                    best_individual = population[best_idx].copy()
                
                best_fitness_history.append(max(fitness_scores))
                
                # Selection, crossover, mutation
                population = self._evolve_population(population, fitness_scores, trains, constraints)
                
                if generation % 20 == 0:
                    logger.info(f"Generation {generation}: Best fitness = {max(fitness_scores):.3f}")
            
            # Convert best individual to schedule
            schedule = self._individual_to_schedule(best_individual, trains)
            
            return {
                "status": "success",
                "schedule": schedule,
                "confidence": min(max(best_fitness_history) / 100, 0.95),  # Normalize to 0-1
                "conflicts_resolved": self._count_conflicts_resolved(schedule),
                "delay_reduction": self._calculate_delay_reduction(schedule, trains),
                "generations": self.generations,
                "best_fitness": max(best_fitness_history),
                "fitness_history": best_fitness_history[-10:]  # Last 10 generations
            }
            
        except Exception as e:
            logger.error(f"GA optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _initialize_population(self, trains, constraints) -> List[List]:
        """Initialize random population of schedules"""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            
            for train in trains:
                # Each gene represents: [departure_offset, platform, priority_boost]
                departure_offset = random.randint(-30, 60)  # -30 to +60 minutes
                platform = random.randint(1, 4)  # 4 platforms
                priority_boost = random.uniform(0, 1)  # Priority adjustment factor
                
                individual.append([departure_offset, platform, priority_boost])
            
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(self, individual, trains, constraints) -> float:
        """
        Evaluate fitness of an individual (schedule)
        Higher fitness = better schedule
        """
        fitness = 100.0  # Start with base fitness
        
        # Convert individual to schedule representation
        schedule_data = []
        for i, gene in enumerate(individual):
            train = trains[i]
            departure_offset, platform, priority_boost = gene
            
            new_departure = train.departure_time + timedelta(minutes=departure_offset)
            new_arrival = train.arrival_time + timedelta(minutes=departure_offset)
            
            schedule_data.append({
                'train': train,
                'departure': new_departure,
                'arrival': new_arrival,
                'platform': platform,
                'priority_boost': priority_boost
            })
        
        # Fitness components
        
        # 1. Minimize delays
        total_delay = 0
        for i, data in enumerate(schedule_data):
            delay = max(0, data['departure_offset'] if 'departure_offset' in data else individual[i][0])
            total_delay += delay * (6 - data['train'].priority)  # Weight by priority
        
        fitness -= total_delay * 0.5  # Penalty for delays
        
        # 2. Avoid platform conflicts
        conflict_penalty = 0
        for i in range(len(schedule_data)):
            for j in range(i + 1, len(schedule_data)):
                if self._has_platform_conflict(schedule_data[i], schedule_data[j]):
                    conflict_penalty += 20
        
        fitness -= conflict_penalty
        
        # 3. Respect priority ordering
        priority_bonus = 0
        for i in range(len(schedule_data)):
            for j in range(i + 1, len(schedule_data)):
                train1 = schedule_data[i]['train']
                train2 = schedule_data[j]['train']
                
                if train1.priority < train2.priority:  # train1 has higher priority
                    if schedule_data[i]['departure'] <= schedule_data[j]['departure']:
                        priority_bonus += 5
        
        fitness += priority_bonus
        
        # 4. Minimize journey time variations
        time_variation_penalty = 0
        for data in schedule_data:
            original_journey = (data['train'].arrival_time - data['train'].departure_time).total_seconds() / 60
            new_journey = (data['arrival'] - data['departure']).total_seconds() / 60
            time_variation_penalty += abs(new_journey - original_journey) * 0.1
        
        fitness -= time_variation_penalty
        
        # 5. Platform utilization efficiency
        platform_usage = {1: 0, 2: 0, 3: 0, 4: 0}
        for data in schedule_data:
            platform_usage[data['platform']] += 1
        
        # Bonus for balanced platform usage
        avg_usage = len(trains) / 4
        balance_bonus = sum(5 - abs(usage - avg_usage) for usage in platform_usage.values())
        fitness += balance_bonus
        
        return max(0, fitness)  # Ensure non-negative fitness
    
    def _has_platform_conflict(self, schedule1, schedule2) -> bool:
        """Check if two trains have platform conflict"""
        if schedule1['platform'] != schedule2['platform']:
            return False
        
        # Check time overlap
        start1, end1 = schedule1['departure'], schedule1['arrival']
        start2, end2 = schedule2['departure'], schedule2['arrival']
        
        return not (end1 < start2 or end2 < start1)
    
    def _evolve_population(self, population, fitness_scores, trains, constraints) -> List[List]:
        """Evolve population through selection, crossover, and mutation"""
        new_population = []
        
        # Elite preservation
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1, trains, constraints)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2, trains, constraints)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3) -> List:
        """Tournament selection for parent selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2) -> Tuple[List, List]:
        """Single-point crossover"""
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, individual, trains, constraints) -> List:
        """Mutate individual with small random changes"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < 0.3:  # 30% chance to mutate each gene
                gene = mutated[i].copy()
                
                # Mutate departure offset
                gene[0] += random.randint(-10, 10)
                gene[0] = max(-30, min(60, gene[0]))  # Clamp to valid range
                
                # Mutate platform (10% chance)
                if random.random() < 0.1:
                    gene[1] = random.randint(1, 4)
                
                # Mutate priority boost
                gene[2] += random.uniform(-0.1, 0.1)
                gene[2] = max(0, min(1, gene[2]))  # Clamp to [0, 1]
                
                mutated[i] = gene
        
        return mutated
    
    def _individual_to_schedule(self, individual, trains) -> List[Dict]:
        """Convert GA individual to schedule format"""
        schedule = []
        
        for i, gene in enumerate(individual):
            train = trains[i]
            departure_offset, platform, priority_boost = gene
            
            optimized_departure = train.departure_time + timedelta(minutes=departure_offset)
            optimized_arrival = train.arrival_time + timedelta(minutes=departure_offset)
            
            schedule.append({
                "train_id": train.train_id,
                "train_number": train.train_number,
                "original_departure": train.departure_time.isoformat(),
                "optimized_departure": optimized_departure.isoformat(),
                "original_arrival": train.arrival_time.isoformat(),
                "optimized_arrival": optimized_arrival.isoformat(),
                "assigned_platform": int(platform),
                "delay_minutes": max(0, departure_offset),
                "priority": train.priority,
                "priority_boost": priority_boost
            })
        
        return schedule
    
    def _count_conflicts_resolved(self, schedule: List[Dict]) -> int:
        """Count conflicts resolved by GA optimization"""
        conflicts = 0
        
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                if schedule[i]['assigned_platform'] == schedule[j]['assigned_platform']:
                    # Check if there was a potential conflict that was resolved
                    time_gap = abs(
                        datetime.fromisoformat(schedule[i]['optimized_departure'].replace('Z', '+00:00')) -
                        datetime.fromisoformat(schedule[j]['optimized_departure'].replace('Z', '+00:00'))
                    ).total_seconds() / 60
                    
                    if time_gap < 30:  # Less than 30 minutes apart - potential conflict
                        conflicts += 1
        
        return conflicts
    
    def _calculate_delay_reduction(self, schedule: List[Dict], original_trains) -> int:
        """Calculate delay reduction achieved by GA"""
        original_total_delay = sum(train.current_delay for train in original_trains)
        optimized_total_delay = sum(max(0, s['delay_minutes']) for s in schedule)
        
        return max(0, original_total_delay - optimized_total_delay)