"""
Main AI Scheduler for Railway Decision Support System

This module provides the core AI scheduling functionality using
4 specialized algorithms for different railway problems:

1. Train Scheduling Conflicts → OR-Tools Scheduler (Constraint Programming)
2. Dynamic Delays & Disruptions → Multi-Agent Reinforcement Learning (MARL)
3. Manual Prioritization at Junctions → Reinforcement Learning
4. Resource Optimization → Genetic Algorithms
5. Maintenance Scheduling → OR-Tools Scheduler
6. High Controller Overload → Zone-based Multi-Agent Reinforcement Learning
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json

from .optimization.or_tools_scheduler import ORToolsScheduler
from .heuristics.genetic_algorithm import GeneticAlgorithmScheduler
from .heuristics.reinforcement_learning import RLScheduler
from .heuristics.multi_agent_rl import MARLScheduler
from .problem_classifier import ProblemClassifier, ProblemType, AlgorithmType

logger = logging.getLogger(__name__)

class Train:
    """Simple Train data class for AI engine"""
    def __init__(self, train_id: int, train_number: str, train_type: str, 
                 priority: int, current_delay: int, source: str, destination: str,
                 departure_time: datetime, arrival_time: datetime):
        self.train_id = train_id
        self.train_number = train_number
        self.train_type = train_type
        self.priority = priority
        self.current_delay = current_delay
        self.source = source
        self.destination = destination
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        
        # Additional attributes for enhanced problem detection
        self.route = getattr(self, 'route', [source, destination])
        self.current_station = getattr(self, 'current_station', source)
        self.assigned_platform = getattr(self, 'assigned_platform', 1)
        self.current_zone = getattr(self, 'current_zone', 'DEFAULT_ZONE')

class AIScheduler:
    """
    Main AI Scheduler class that coordinates 4 specialized optimization algorithms
    based on intelligent problem classification
    """
    
    def __init__(self):
        """Initialize AI Scheduler with all 4 optimization engines"""
        # Initialize all 4 algorithms
        self.or_tools_scheduler = ORToolsScheduler()
        self.genetic_scheduler = GeneticAlgorithmScheduler()
        self.rl_scheduler = RLScheduler()
        self.marl_scheduler = MARLScheduler()
        
        # Initialize problem classifier
        self.problem_classifier = ProblemClassifier()
        
        # Algorithm performance tracking
        self.algorithm_performance = {
            AlgorithmType.OR_TOOLS: {"calls": 0, "successes": 0, "avg_confidence": 0.0},
            AlgorithmType.GENETIC_ALGORITHM: {"calls": 0, "successes": 0, "avg_confidence": 0.0},
            AlgorithmType.REINFORCEMENT_LEARNING: {"calls": 0, "successes": 0, "avg_confidence": 0.0},
            AlgorithmType.MULTI_AGENT_RL: {"calls": 0, "successes": 0, "avg_confidence": 0.0}
        }
        
        # Configuration
        self.optimization_timeout = 30  # seconds
        self.max_iterations = 1000
        self.enable_multi_algorithm = True  # Allow multiple algorithms for complex problems
        
        logger.info("AI Scheduler initialized with 4 algorithms: OR-Tools, GA, RL, and MARL")
    
    def optimize_schedule(self, trains: List[Dict], constraints: Dict[str, Any] = None) -> Dict:
        """
        Main schedule optimization method with intelligent problem classification
        
        Args:
            trains: List of train dictionaries with train data
            constraints: Dictionary of optimization constraints
            
        Returns:
            Optimized schedule with recommendations from appropriate algorithms
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert train dictionaries to Train objects
            train_objects = self._convert_to_train_objects(trains)
            
            if not train_objects:
                return self._create_empty_response("No trains provided")
            
            logger.info(f"Starting intelligent optimization for {len(train_objects)} trains")
            
            # Step 1: Classify problems to determine which algorithms to use
            detected_problems = self.problem_classifier.classify_problems(train_objects, constraints)
            
            if not detected_problems:
                # No specific problems detected, use OR-Tools as default
                logger.info("No specific problems detected, using OR-Tools as default")
                return self._run_single_algorithm(AlgorithmType.OR_TOOLS, train_objects, constraints)
            
            # Step 2: Prioritize problems and select algorithms
            prioritized_problems = self.problem_classifier.get_problem_priority(detected_problems)
            
            # Step 3: Run appropriate algorithms for each problem
            algorithm_results = {}
            algorithms_used = set()
            
            for problem_type, problem_data in prioritized_problems:
                algorithm = self.problem_classifier.get_algorithm_for_problem(problem_type)
                algorithms_used.add(algorithm)
                
                logger.info(f"Solving {problem_type.value} using {algorithm.value}")
                
                # Run algorithm for this specific problem
                result = self._run_algorithm_for_problem(algorithm, train_objects, constraints, problem_data)
                algorithm_results[problem_type] = {
                    "algorithm": algorithm,
                    "result": result,
                    "problem_data": problem_data
                }
            
            # Step 4: Coordinate and merge results if multiple algorithms were used
            if len(algorithms_used) > 1:
                final_result = self._coordinate_multiple_algorithms(algorithm_results, train_objects, constraints)
            else:
                # Single algorithm used
                primary_result = list(algorithm_results.values())[0]["result"]
                final_result = self._enhance_single_result(primary_result, detected_problems, algorithms_used)
            
            # Step 5: Add meta information about the optimization process
            final_result["optimization_meta"] = {
                "problems_detected": [pt.value for pt, _ in prioritized_problems],
                "algorithms_used": [alg.value for alg in algorithms_used],
                "total_processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "coordination_complexity": len(algorithms_used),
                "problem_severity": self._calculate_overall_severity(detected_problems)
            }
            
            # Update algorithm performance tracking
            self._update_algorithm_performance(algorithms_used, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"AI optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _run_algorithm_for_problem(self, algorithm: AlgorithmType, train_objects: List, 
                                 constraints: Dict, problem_data: Dict) -> Dict:
        """Run specific algorithm for a classified problem"""
        try:
            if algorithm == AlgorithmType.OR_TOOLS:
                return self.or_tools_scheduler.optimize(train_objects, constraints)
            elif algorithm == AlgorithmType.GENETIC_ALGORITHM:
                return self.genetic_scheduler.optimize(train_objects, constraints)
            elif algorithm == AlgorithmType.REINFORCEMENT_LEARNING:
                return self.rl_scheduler.optimize(train_objects, constraints)
            elif algorithm == AlgorithmType.MULTI_AGENT_RL:
                return self.marl_scheduler.optimize(train_objects, constraints)
            else:
                logger.warning(f"Unknown algorithm type: {algorithm}")
                return self.or_tools_scheduler.optimize(train_objects, constraints)
                
        except Exception as e:
            logger.error(f"Algorithm {algorithm.value} failed: {e}")
            return {"status": "failed", "error": str(e), "algorithm": algorithm.value}
    
    def _coordinate_multiple_algorithms(self, algorithm_results: Dict, train_objects: List, 
                                      constraints: Dict) -> Dict:
        """Coordinate results from multiple algorithms for complex problems"""
        logger.info(f"Coordinating results from {len(algorithm_results)} algorithms")
        
        # Collect all successful results
        successful_results = {}
        all_schedules = []
        
        for problem_type, result_data in algorithm_results.items():
            result = result_data["result"]
            if result.get("status") == "success":
                successful_results[problem_type] = result_data
                if "schedule" in result:
                    all_schedules.extend(result["schedule"])
        
        if not successful_results:
            return {"status": "failed", "error": "All algorithms failed"}
        
        # Strategy 1: If MARL was successful, use it as primary (it handles coordination)
        marl_result = None
        for problem_type, result_data in successful_results.items():
            if result_data["algorithm"] == AlgorithmType.MULTI_AGENT_RL:
                marl_result = result_data
                break
        
        if marl_result:
            logger.info("Using MARL as primary coordinator")
            primary_result = marl_result["result"]
            
            # Enhance with insights from other algorithms
            enhancements = self._extract_enhancements_from_other_algorithms(
                successful_results, marl_result
            )
            primary_result["coordination_enhancements"] = enhancements
            
        else:
            # Strategy 2: Merge results using weighted approach
            logger.info("Merging results using weighted approach")
            primary_result = self._merge_algorithm_results(successful_results, train_objects)
        
        # Add coordination metadata
        primary_result["multi_algorithm_coordination"] = {
            "algorithms_coordinated": [rd["algorithm"].value for rd in successful_results.values()],
            "coordination_strategy": "marl_primary" if marl_result else "weighted_merge",
            "problems_addressed": list(successful_results.keys()),
            "coordination_confidence": self._calculate_coordination_confidence(successful_results)
        }
        
        return primary_result
    
    def _merge_algorithm_results(self, successful_results: Dict, train_objects: List) -> Dict:
        """Merge results from multiple algorithms using weighted approach"""
        # Algorithm weights based on confidence and problem relevance
        algorithm_weights = {
            AlgorithmType.OR_TOOLS: 0.3,
            AlgorithmType.GENETIC_ALGORITHM: 0.25,
            AlgorithmType.REINFORCEMENT_LEARNING: 0.2,
            AlgorithmType.MULTI_AGENT_RL: 0.25
        }
        
        merged_schedule = []
        confidence_scores = []
        
        # Create train-specific schedule by averaging recommendations
        for train in train_objects:
            train_recommendations = []
            
            for problem_type, result_data in successful_results.items():
                result = result_data["result"]
                algorithm = result_data["algorithm"]
                
                if "schedule" in result:
                    for schedule_entry in result["schedule"]:
                        if schedule_entry.get("train_id") == train.train_id:
                            train_recommendations.append({
                                "algorithm": algorithm,
                                "weight": algorithm_weights.get(algorithm, 0.2),
                                "entry": schedule_entry,
                                "confidence": result.get("confidence", 0.5)
                            })
            
            if train_recommendations:
                # Merge recommendations for this train
                merged_entry = self._merge_train_recommendations(train, train_recommendations)
                merged_schedule.append(merged_entry)
                
                # Track confidence
                weighted_confidence = sum(
                    rec["confidence"] * rec["weight"] for rec in train_recommendations
                ) / sum(rec["weight"] for rec in train_recommendations)
                confidence_scores.append(weighted_confidence)
        
        # Calculate overall metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        return {
            "status": "success",
            "schedule": merged_schedule,
            "confidence": avg_confidence,
            "merge_strategy": "weighted_average",
            "algorithms_merged": [rd["algorithm"].value for rd in successful_results.values()],
            "coordination_complexity": len(successful_results)
        }
    
    def _merge_train_recommendations(self, train, recommendations: List[Dict]) -> Dict:
        """Merge multiple algorithm recommendations for a single train"""
        # Base schedule entry
        base_entry = {
            "train_id": train.train_id,
            "train_number": train.train_number,
            "original_departure": train.departure_time.isoformat(),
            "original_arrival": train.arrival_time.isoformat(),
            "priority": train.priority
        }
        
        # Weighted averages for numerical values
        total_weight = sum(rec["weight"] for rec in recommendations)
        
        # Merge delay adjustments
        delay_adjustments = []
        platform_assignments = []
        
        for rec in recommendations:
            entry = rec["entry"]
            weight = rec["weight"]
            
            if "delay_minutes" in entry:
                delay_adjustments.append(entry["delay_minutes"] * weight)
            if "assigned_platform" in entry:
                platform_assignments.append(entry["assigned_platform"])
        
        # Calculate weighted average delay
        if delay_adjustments:
            avg_delay = sum(delay_adjustments) / total_weight
            base_entry["delay_minutes"] = max(0, int(avg_delay))
        else:
            base_entry["delay_minutes"] = train.current_delay
        
        # Select most common platform assignment
        if platform_assignments:
            platform_counts = {}
            for platform in platform_assignments:
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
            base_entry["assigned_platform"] = max(platform_counts, key=platform_counts.get)
        else:
            base_entry["assigned_platform"] = getattr(train, 'assigned_platform', 1)
        
        # Calculate optimized times
        delay_delta = timedelta(minutes=base_entry["delay_minutes"] - train.current_delay)
        base_entry["optimized_departure"] = (train.departure_time + delay_delta).isoformat()
        base_entry["optimized_arrival"] = (train.arrival_time + delay_delta).isoformat()
        
        # Add algorithm attribution
        base_entry["contributing_algorithms"] = [rec["algorithm"].value for rec in recommendations]
        
        return base_entry
    
    def _run_single_algorithm(self, algorithm: AlgorithmType, train_objects: List, constraints: Dict) -> Dict:
        """Run a single algorithm when no specific problems are detected"""
        return self._run_algorithm_for_problem(algorithm, train_objects, constraints, {})
    
    def _enhance_single_result(self, result: Dict, detected_problems: Dict, algorithms_used: set) -> Dict:
        """Enhance result when only one algorithm was used"""
        result["single_algorithm_optimization"] = {
            "primary_algorithm": list(algorithms_used)[0].value,
            "problems_detected": [pt.value for pt in detected_problems.keys()],
            "optimization_focus": "targeted" if detected_problems else "general"
        }
        return result
    
    def _extract_enhancements_from_other_algorithms(self, successful_results: Dict, marl_result: Dict) -> Dict:
        """Extract insights from other algorithms to enhance MARL results"""
        enhancements = {}
        
        for problem_type, result_data in successful_results.items():
            if result_data["algorithm"] != AlgorithmType.MULTI_AGENT_RL:
                algorithm = result_data["algorithm"]
                result = result_data["result"]
                
                enhancements[algorithm.value] = {
                    "confidence": result.get("confidence", 0.5),
                    "key_insights": self._extract_algorithm_insights(algorithm, result),
                    "complementary_recommendations": self._get_complementary_recommendations(algorithm, result)
                }
        
        return enhancements
    
    def _extract_algorithm_insights(self, algorithm: AlgorithmType, result: Dict) -> List[str]:
        """Extract key insights from algorithm results"""
        insights = []
        
        if algorithm == AlgorithmType.OR_TOOLS:
            if result.get("conflicts_resolved", 0) > 0:
                insights.append(f"Resolved {result['conflicts_resolved']} scheduling conflicts")
            if "constraint_satisfaction" in result:
                insights.append(f"Constraint satisfaction: {result['constraint_satisfaction']}%")
                
        elif algorithm == AlgorithmType.GENETIC_ALGORITHM:
            if result.get("fitness_improvement", 0) > 0:
                insights.append(f"Fitness improvement: {result['fitness_improvement']}")
            if "resource_efficiency" in result:
                insights.append(f"Resource efficiency: {result['resource_efficiency']}%")
                
        elif algorithm == AlgorithmType.REINFORCEMENT_LEARNING:
            if result.get("exploration_rate"):
                insights.append(f"RL exploration rate: {result['exploration_rate']}")
            if result.get("q_table_size"):
                insights.append(f"Knowledge base size: {result['q_table_size']} states")
        
        return insights
    
    def _get_complementary_recommendations(self, algorithm: AlgorithmType, result: Dict) -> List[str]:
        """Get complementary recommendations from other algorithms"""
        recommendations = []
        
        if algorithm == AlgorithmType.OR_TOOLS and result.get("confidence", 0) > 0.8:
            recommendations.append("High-confidence constraint optimization available")
            
        elif algorithm == AlgorithmType.GENETIC_ALGORITHM and result.get("resource_efficiency", 0) > 80:
            recommendations.append("Significant resource optimization potential identified")
            
        elif algorithm == AlgorithmType.REINFORCEMENT_LEARNING and result.get("reward", 0) > 0:
            recommendations.append("Positive learning outcomes for future decisions")
        
        return recommendations
    
    def _calculate_coordination_confidence(self, successful_results: Dict) -> float:
        """Calculate confidence score for multi-algorithm coordination"""
        confidences = []
        weights = []
        
        for result_data in successful_results.values():
            result = result_data["result"]
            algorithm = result_data["algorithm"]
            
            confidence = result.get("confidence", 0.5)
            
            # Weight based on algorithm type
            weight = {
                AlgorithmType.OR_TOOLS: 1.2,  # High weight for constraint satisfaction
                AlgorithmType.MULTI_AGENT_RL: 1.1,  # High weight for coordination
                AlgorithmType.GENETIC_ALGORITHM: 1.0,
                AlgorithmType.REINFORCEMENT_LEARNING: 0.9
            }.get(algorithm, 1.0)
            
            confidences.append(confidence)
            weights.append(weight)
        
        if not confidences:
            return 0.5
        
        # Weighted average confidence
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        # Bonus for successful coordination
        coordination_bonus = min(0.1, len(successful_results) * 0.02)
        
        return min(1.0, weighted_confidence + coordination_bonus)
    
    def _calculate_overall_severity(self, detected_problems: Dict) -> str:
        """Calculate overall severity across all detected problems"""
        severities = [problem_data.get("severity", "medium") for problem_data in detected_problems.values()]
        
        if "critical" in severities:
            return "critical"
        elif severities.count("high") >= 2:
            return "high"
        elif "high" in severities:
            return "high"
        elif "medium" in severities:
            return "medium"
        else:
            return "low"
    
    def _update_algorithm_performance(self, algorithms_used: set, result: Dict):
        """Update performance tracking for algorithms"""
        success = result.get("status") == "success"
        confidence = result.get("confidence", 0.5)
        
        for algorithm in algorithms_used:
            stats = self.algorithm_performance[algorithm]
            stats["calls"] += 1
            
            if success:
                stats["successes"] += 1
            
            # Update average confidence
            current_avg = stats["avg_confidence"]
            call_count = stats["calls"]
            stats["avg_confidence"] = (current_avg * (call_count - 1) + confidence) / call_count
    
    def get_algorithm_performance(self) -> Dict:
        """Get performance statistics for all algorithms"""
        performance_summary = {}
        
        for algorithm, stats in self.algorithm_performance.items():
            calls = stats["calls"]
            successes = stats["successes"]
            
            performance_summary[algorithm.value] = {
                "total_calls": calls,
                "successful_calls": successes,
                "success_rate": (successes / calls * 100) if calls > 0 else 0,
                "average_confidence": round(stats["avg_confidence"], 3),
                "reliability_score": (successes / calls * stats["avg_confidence"]) if calls > 0 else 0
            }
        
        return performance_summary
    
    def resolve_conflict(self, train1: Dict, train2: Dict, location: str, 
                        conflict_type: str = "precedence") -> Dict:
        """
        Resolve conflict between two trains
        
        Args:
            train1: First train data
            train2: Second train data
            location: Conflict location
            conflict_type: Type of conflict (precedence, crossing, platform)
            
        Returns:
            Conflict resolution decision
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert to Train objects
            t1 = self._dict_to_train(train1)
            t2 = self._dict_to_train(train2)
            
            # Use OR-Tools for conflict resolution
            resolution = self.or_tools_scheduler.resolve_conflict(t1, t2, location, conflict_type)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "status": "success",
                "conflict_type": conflict_type,
                "location": location,
                "resolution": resolution,
                "execution_time_seconds": execution_time,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Conflict resolution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": start_time.isoformat()
            }
    
    # Helper methods for the new multi-algorithm system
    
    def _convert_to_train_objects(self, trains: List[Dict]) -> List[Train]:
        """Convert train dictionaries to Train objects"""
        train_objects = []
        
        for train_data in trains:
            try:
                # Parse datetime strings if they are strings
                departure = train_data.get('departure_time')
                arrival = train_data.get('arrival_time')
                
                if isinstance(departure, str):
                    departure = datetime.fromisoformat(departure.replace('Z', '+00:00'))
                if isinstance(arrival, str):
                    arrival = datetime.fromisoformat(arrival.replace('Z', '+00:00'))
                
                train = Train(
                    train_id=train_data.get('train_id', train_data.get('id')),
                    train_number=train_data.get('train_number', ''),
                    train_type=train_data.get('train_type', 'PASSENGER'),
                    priority=train_data.get('priority', 3),
                    current_delay=train_data.get('current_delay', 0),
                    source=train_data.get('source', train_data.get('origin', '')),
                    destination=train_data.get('destination', ''),
                    departure_time=departure or datetime.utcnow(),
                    arrival_time=arrival or datetime.utcnow() + timedelta(hours=2)
                )
                
                # Add additional attributes for enhanced problem detection
                train.route = train_data.get('route', [train.source, train.destination])
                train.current_station = train_data.get('current_station', train.source)
                train.assigned_platform = train_data.get('assigned_platform', 1)
                train.current_zone = train_data.get('current_zone', 'DEFAULT_ZONE')
                
                train_objects.append(train)
                
            except Exception as e:
                logger.warning(f"Failed to convert train data: {e}")
                continue
        
        return train_objects
    
    def _create_empty_response(self, message: str) -> Dict:
        """Create empty response for edge cases"""
        return {
            "status": "success",
            "message": message,
            "schedule": [],
            "confidence": 1.0,
            "optimization_meta": {
                "problems_detected": [],
                "algorithms_used": [],
                "total_processing_time_ms": 0,
                "coordination_complexity": 0,
                "problem_severity": "none"
            }
        }
    
    def generate_recommendations(self, current_situation: Dict) -> List[Dict]:
        """
        Generate AI recommendations for current railway situation
        
        Args:
            current_situation: Current state of trains and infrastructure
            
        Returns:
            List of AI recommendations
        """
        try:
            recommendations = []
            
            trains = current_situation.get('trains', [])
            delays = current_situation.get('delays', {})
            conflicts = current_situation.get('conflicts', [])
            
            # Analyze delays
            for train in trains:
                delay = train.get('current_delay', 0)
                if delay > 15:
                    recommendations.append({
                        "type": "DELAY_MANAGEMENT",
                        "priority": "HIGH",
                        "train_id": train['id'],
                        "message": f"Train {train['train_number']} has {delay} min delay - consider speed increase or route optimization",
                        "confidence": 0.85,
                        "estimated_benefit": f"{min(delay//2, 10)} minutes recovery"
                    })
            
            # Analyze conflicts
            for conflict in conflicts:
                recommendations.append({
                    "type": "CONFLICT_RESOLUTION",
                    "priority": "CRITICAL",
                    "message": f"Resolve {conflict['type']} conflict at {conflict['location']}",
                    "confidence": 0.92,
                    "estimated_benefit": "Prevent cascading delays"
                })
            
            # System-level recommendations
            delayed_count = len([t for t in trains if t.get('current_delay', 0) > 5])
            if delayed_count > len(trains) * 0.3:  # More than 30% delayed
                recommendations.append({
                    "type": "SYSTEM_OPTIMIZATION",
                    "priority": "HIGH",
                    "message": "High system delay detected - implement dynamic rescheduling",
                    "confidence": 0.78,
                    "estimated_benefit": "20-30% delay reduction"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def predict_delays(self, trains: List[Dict], weather: Dict = None, 
                      infrastructure: Dict = None) -> Dict:
        """
        Predict potential delays based on current conditions
        
        Args:
            trains: List of train data
            weather: Weather conditions
            infrastructure: Infrastructure status
            
        Returns:
            Delay predictions
        """
        try:
            predictions = []
            
            for train in trains:
                base_delay = train.get('current_delay', 0)
                predicted_delay = base_delay
                
                # Weather impact
                if weather:
                    weather_condition = weather.get('condition', 'clear')
                    if weather_condition == 'heavy_rain':
                        predicted_delay += 5
                    elif weather_condition == 'fog':
                        predicted_delay += 10
                    elif weather_condition == 'storm':
                        predicted_delay += 15
                
                # Infrastructure impact
                if infrastructure:
                    issues = infrastructure.get('issues', [])
                    predicted_delay += len(issues) * 3  # 3 min per issue
                
                # Train type impact
                if train.get('train_type') == 'FREIGHT':
                    predicted_delay += 2  # Freight tends to have more delays
                
                predictions.append({
                    "train_id": train['id'],
                    "train_number": train['train_number'],
                    "current_delay": base_delay,
                    "predicted_delay": predicted_delay,
                    "delay_increase": predicted_delay - base_delay,
                    "confidence": 0.75
                })
            
            return {
                "status": "success",
                "predictions": predictions,
                "avg_delay_increase": sum(p['delay_increase'] for p in predictions) / len(predictions) if predictions else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Delay prediction failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    # Helper methods
    
    def _convert_to_train_objects(self, trains: List[Dict]) -> List[Train]:
        """Convert train dictionaries to Train objects"""
        train_objects = []
        for train_data in trains:
            try:
                train = self._dict_to_train(train_data)
                train_objects.append(train)
            except Exception as e:
                logger.warning(f"Failed to convert train {train_data.get('id', 'unknown')}: {e}")
        return train_objects
    
    def _dict_to_train(self, train_data: Dict) -> Train:
        """Convert dictionary to Train object"""
        return Train(
            train_id=train_data['id'],
            train_number=train_data['train_number'],
            train_type=train_data.get('train_type', 'PASSENGER'),
            priority=train_data.get('priority', 3),
            current_delay=train_data.get('current_delay', 0),
            source=train_data.get('source_station', 'UNKNOWN'),
            destination=train_data.get('destination_station', 'UNKNOWN'),
            departure_time=datetime.fromisoformat(train_data['departure_time'].replace('Z', '+00:00')) if isinstance(train_data.get('departure_time'), str) else train_data.get('departure_time', datetime.utcnow()),
            arrival_time=datetime.fromisoformat(train_data['arrival_time'].replace('Z', '+00:00')) if isinstance(train_data.get('arrival_time'), str) else train_data.get('arrival_time', datetime.utcnow() + timedelta(hours=2))
        )
    
    def _get_default_constraints(self) -> Dict:
        """Get default optimization constraints"""
        return {
            "max_delay_minutes": 30,
            "min_buffer_minutes": 5,
            "priority_weights": {1: 10, 2: 8, 3: 5, 4: 3, 5: 1},
            "track_capacity": 2,
            "platform_capacity": 4,
            "max_speed_kmh": 120
        }
    
    def _select_best_result(self, results: Dict) -> Dict:
        """Select best optimization result from multiple methods"""
        
        # Priority order: OR-Tools > Genetic Algorithm
        for method in ['or_tools', 'genetic_algorithm']:
            if method in results and results[method].get('status') == 'success':
                result = results[method]
                result['method'] = method
                return result
        
        # If all failed, return a fallback result
        return {
            "method": "fallback",
            "status": "fallback",
            "schedule": [],
            "confidence": 0.0,
            "conflicts_resolved": 0,
            "delay_reduction": 0,
            "recommendations": ["All optimization methods failed - manual intervention required"]
        }
    
    def _create_empty_response(self, message: str) -> Dict:
        """Create empty response for edge cases"""
        return {
            "status": "success",
            "message": message,
            "optimized_schedule": [],
            "confidence_score": 0.0,
            "execution_time_seconds": 0.0,
            "total_trains": 0,
            "conflicts_resolved": 0,
            "delay_reduction_minutes": 0,
            "recommendations": [],
            "timestamp": datetime.utcnow().isoformat()
        }