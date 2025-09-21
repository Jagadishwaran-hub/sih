"""
Main AI Scheduler for Railway Decision Support System

This module provides the core AI scheduling functionality using
optimization algorithms and heuristics for train management.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import json

from .optimization.or_tools_scheduler import ORToolsScheduler
from .heuristics.genetic_algorithm import GeneticAlgorithmScheduler
from .heuristics.reinforcement_learning import RLScheduler

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

class AIScheduler:
    """
    Main AI Scheduler class that coordinates different optimization approaches
    """
    
    def __init__(self):
        """Initialize AI Scheduler with optimization engines"""
        self.or_tools_scheduler = ORToolsScheduler()
        self.genetic_scheduler = GeneticAlgorithmScheduler()
        self.rl_scheduler = RLScheduler()
        
        # Configuration
        self.optimization_timeout = 30  # seconds
        self.max_iterations = 1000
        
        logger.info("AI Scheduler initialized with OR-Tools, GA, and RL engines")
    
    def optimize_schedule(self, trains: List[Dict], constraints: Dict[str, Any] = None) -> Dict:
        """
        Main schedule optimization method
        
        Args:
            trains: List of train dictionaries with train data
            constraints: Dictionary of optimization constraints
            
        Returns:
            Optimized schedule with recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            # Convert train dictionaries to Train objects
            train_objects = self._convert_to_train_objects(trains)
            
            if not train_objects:
                return self._create_empty_response("No trains provided")
            
            logger.info(f"Optimizing schedule for {len(train_objects)} trains")
            
            # Apply default constraints if none provided
            if constraints is None:
                constraints = self._get_default_constraints()
            
            # Run different optimization approaches
            results = {}
            
            # 1. OR-Tools optimization (primary)
            try:
                or_result = self.or_tools_scheduler.optimize(train_objects, constraints)
                results['or_tools'] = or_result
                logger.info("OR-Tools optimization completed")
            except Exception as e:
                logger.error(f"OR-Tools optimization failed: {e}")
                results['or_tools'] = {"status": "failed", "error": str(e)}
            
            # 2. Genetic Algorithm (secondary)
            try:
                ga_result = self.genetic_scheduler.optimize(train_objects, constraints)
                results['genetic_algorithm'] = ga_result
                logger.info("Genetic Algorithm optimization completed")
            except Exception as e:
                logger.error(f"GA optimization failed: {e}")
                results['genetic_algorithm'] = {"status": "failed", "error": str(e)}
            
            # 3. Select best result
            best_result = self._select_best_result(results)
            
            # Add execution metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = {
                "status": "success",
                "method_used": best_result['method'],
                "optimized_schedule": best_result['schedule'],
                "confidence_score": best_result['confidence'],
                "execution_time_seconds": execution_time,
                "total_trains": len(train_objects),
                "conflicts_resolved": best_result.get('conflicts_resolved', 0),
                "delay_reduction_minutes": best_result.get('delay_reduction', 0),
                "recommendations": best_result.get('recommendations', []),
                "timestamp": start_time.isoformat()
            }
            
            logger.info(f"Schedule optimization completed in {execution_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Schedule optimization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "timestamp": start_time.isoformat()
            }
    
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