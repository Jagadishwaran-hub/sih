"""
Simulation API endpoints for what-if scenarios
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
import json
import uuid

from ..models.database import Train
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class SimulationScenario(BaseModel):
    scenario_name: str
    description: str
    train_delays: Dict[int, int] = {}  # train_id: delay_minutes
    infrastructure_issues: List[Dict[str, Any]] = []
    weather_conditions: Dict[str, Any] = {}
    duration_hours: int = 24

class SimulationResult(BaseModel):
    simulation_id: str
    scenario_name: str
    status: str
    start_time: datetime
    end_time: datetime = None
    results: Dict[str, Any] = {}

# In-memory simulation store (would use Redis in production)
active_simulations: Dict[str, Dict] = {}

@router.post("/scenarios/run")
async def run_simulation_scenario(scenario: SimulationScenario):
    """
    Run a what-if simulation scenario
    
    This endpoint creates and runs a simulation based on the provided
    scenario parameters including delays, infrastructure issues, and weather.
    """
    simulation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Get trains for simulation
        all_trains = await Train.find(Train.is_active == True).to_list()
        
        # Initialize simulation
        simulation_data = {
            "id": simulation_id,
            "scenario": scenario.dict(),
            "status": "RUNNING",
            "start_time": start_time,
            "trains": [{"id": str(t.id), "number": t.train_number, "name": t.train_name} for t in all_trains]
        }
        
        active_simulations[simulation_id] = simulation_data
        
        # Run simulation (dummy implementation)
        results = await _run_dummy_simulation(scenario, all_trains)
        
        # Update simulation with results
        simulation_data["status"] = "COMPLETED"
        simulation_data["end_time"] = datetime.utcnow()
        simulation_data["results"] = results
        
        logger.info(f"Simulation {simulation_id} completed for scenario: {scenario.scenario_name}")
        
        return {
            "simulation_id": simulation_id,
            "status": "COMPLETED",
            "scenario_name": scenario.scenario_name,
            "execution_time": (simulation_data["end_time"] - start_time).total_seconds(),
            "results": results
        }
        
    except Exception as e:
        if simulation_id in active_simulations:
            active_simulations[simulation_id]["status"] = "FAILED"
            active_simulations[simulation_id]["error"] = str(e)
        
        logger.error(f"Simulation {simulation_id} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/scenarios/{simulation_id}")
async def get_simulation_results(simulation_id: str):
    """Get results of a specific simulation"""
    
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    simulation = active_simulations[simulation_id]
    
    return {
        "simulation_id": simulation_id,
        "scenario_name": simulation["scenario"]["scenario_name"],
        "status": simulation["status"],
        "start_time": simulation["start_time"].isoformat(),
        "end_time": simulation.get("end_time", {}).isoformat() if simulation.get("end_time") else None,
        "results": simulation.get("results", {}),
        "error": simulation.get("error")
    }

@router.get("/scenarios")
async def list_simulations():
    """List all simulations (active and completed)"""
    
    simulations = []
    for sim_id, sim_data in active_simulations.items():
        simulations.append({
            "simulation_id": sim_id,
            "scenario_name": sim_data["scenario"]["scenario_name"],
            "status": sim_data["status"],
            "start_time": sim_data["start_time"].isoformat(),
            "train_count": len(sim_data["trains"])
        })
    
    return {
        "total_simulations": len(simulations),
        "simulations": simulations
    }

@router.post("/scenarios/compare")
async def compare_scenarios(
    simulation_ids: List[str]
):
    """Compare results from multiple simulation scenarios"""
    
    if len(simulation_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 simulations required for comparison")
    
    comparison_results = []
    
    for sim_id in simulation_ids:
        if sim_id not in active_simulations:
            raise HTTPException(status_code=404, detail=f"Simulation {sim_id} not found")
        
        sim = active_simulations[sim_id]
        if sim["status"] != "COMPLETED":
            raise HTTPException(status_code=400, detail=f"Simulation {sim_id} not completed")
        
        comparison_results.append({
            "simulation_id": sim_id,
            "scenario_name": sim["scenario"]["scenario_name"],
            "results": sim["results"]
        })
    
    # Generate comparison analysis
    comparison = _generate_scenario_comparison(comparison_results)
    
    return {
        "comparison_id": str(uuid.uuid4()),
        "compared_scenarios": len(simulation_ids),
        "scenarios": comparison_results,
        "analysis": comparison,
        "generated_at": datetime.utcnow().isoformat()
    }

@router.delete("/scenarios/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Delete a simulation from memory"""
    
    if simulation_id not in active_simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del active_simulations[simulation_id]
    
    return {"message": f"Simulation {simulation_id} deleted successfully"}

@router.post("/scenarios/presets/delay")
async def create_delay_scenario(
    delay_minutes: int,
    affected_train_ids: List[str]
):
    """Create a preset delay scenario for quick testing"""
    
    scenario = SimulationScenario(
        scenario_name=f"Delay Test - {delay_minutes} minutes",
        description=f"Testing impact of {delay_minutes} minute delay on {len(affected_train_ids)} trains",
        train_delays={train_id: delay_minutes for train_id in affected_train_ids},
        duration_hours=12
    )
    
    return await run_simulation_scenario(scenario)

# Helper functions for simulation logic

async def _run_dummy_simulation(scenario: SimulationScenario, trains: List[Train]):
    """Run dummy simulation logic"""
    
    # Calculate baseline metrics
    baseline_avg_delay = sum(t.current_delay for t in trains) / len(trains) if trains else 0
    baseline_on_time = len([t for t in trains if t.current_delay <= 5]) / len(trains) * 100 if trains else 0
    
    # Apply scenario modifications
    total_additional_delay = sum(scenario.train_delays.values())
    affected_trains = len(scenario.train_delays)
    
    # Calculate new metrics after scenario
    new_avg_delay = baseline_avg_delay + (total_additional_delay / len(trains) if trains else 0)
    new_on_time = max(0, baseline_on_time - (affected_trains / len(trains) * 20 if trains else 0))
    
    # Infrastructure impact
    infrastructure_delay = len(scenario.infrastructure_issues) * 5  # 5 min per issue
    
    # Weather impact
    weather_impact = 0
    if scenario.weather_conditions.get("condition") == "heavy_rain":
        weather_impact = 10
    elif scenario.weather_conditions.get("condition") == "fog":
        weather_impact = 15
    
    final_avg_delay = new_avg_delay + infrastructure_delay + weather_impact
    final_on_time = max(0, new_on_time - (infrastructure_delay + weather_impact) / 5)
    
    results = {
        "baseline_metrics": {
            "average_delay_minutes": round(baseline_avg_delay, 2),
            "on_time_percentage": round(baseline_on_time, 2),
            "total_trains": len(trains)
        },
        "scenario_impact": {
            "additional_train_delays": total_additional_delay,
            "affected_trains": affected_trains,
            "infrastructure_delay": infrastructure_delay,
            "weather_delay": weather_impact
        },
        "final_metrics": {
            "average_delay_minutes": round(final_avg_delay, 2),
            "on_time_percentage": round(final_on_time, 2),
            "delay_increase": round(final_avg_delay - baseline_avg_delay, 2),
            "punctuality_decrease": round(baseline_on_time - final_on_time, 2)
        },
        "performance_impact": {
            "delay_change_percent": round(((final_avg_delay - baseline_avg_delay) / baseline_avg_delay * 100) if baseline_avg_delay > 0 else 0, 2),
            "punctuality_change_percent": round(((final_on_time - baseline_on_time) / baseline_on_time * 100) if baseline_on_time > 0 else 0, 2)
        },
        "recommendations": _generate_simulation_recommendations(final_avg_delay, final_on_time)
    }
    
    return results

def _generate_simulation_recommendations(avg_delay: float, on_time_percent: float):
    """Generate recommendations based on simulation results"""
    
    recommendations = []
    
    if avg_delay > 15:
        recommendations.append({
            "type": "CRITICAL",
            "message": "Consider implementing dynamic rescheduling",
            "impact": "Could reduce delays by 20-30%"
        })
    
    if on_time_percent < 70:
        recommendations.append({
            "type": "HIGH",
            "message": "Increase buffer times between trains",
            "impact": "Improve punctuality by 15-20%"
        })
    
    if avg_delay > 10:
        recommendations.append({
            "type": "MEDIUM",
            "message": "Deploy additional traffic controllers",
            "impact": "Better conflict resolution"
        })
    
    return recommendations

def _generate_scenario_comparison(scenarios: List[Dict]):
    """Generate comparative analysis of scenarios"""
    
    best_delay = min(s["results"]["final_metrics"]["average_delay_minutes"] for s in scenarios)
    best_punctuality = max(s["results"]["final_metrics"]["on_time_percentage"] for s in scenarios)
    
    analysis = {
        "best_delay_scenario": None,
        "best_punctuality_scenario": None,
        "overall_recommendation": "",
        "key_insights": []
    }
    
    for scenario in scenarios:
        metrics = scenario["results"]["final_metrics"]
        
        if metrics["average_delay_minutes"] == best_delay:
            analysis["best_delay_scenario"] = scenario["scenario_name"]
        
        if metrics["on_time_percentage"] == best_punctuality:
            analysis["best_punctuality_scenario"] = scenario["scenario_name"]
    
    delay_range = max(s["results"]["final_metrics"]["average_delay_minutes"] for s in scenarios) - best_delay
    punctuality_range = best_punctuality - min(s["results"]["final_metrics"]["on_time_percentage"] for s in scenarios)
    
    analysis["key_insights"] = [
        f"Delay variation across scenarios: {delay_range:.1f} minutes",
        f"Punctuality variation: {punctuality_range:.1f}%",
        f"Best performing scenario for delays: {analysis['best_delay_scenario']}",
        f"Best performing scenario for punctuality: {analysis['best_punctuality_scenario']}"
    ]
    
    return analysis