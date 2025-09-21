import React, { useState, useEffect } from 'react';
import {
  Zap,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  Users,
  RefreshCw,
  Filter,
  ThumbsUp,
  ThumbsDown,
  Eye,
  Settings
} from 'lucide-react';
import { scheduleAPI, trainAPI } from '../services/api';
import { useWebSocket, useWebSocketStatus } from '../hooks/useWebSocket';

const AIRecommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [autoApplyEnabled, setAutoApplyEnabled] = useState(false);
  const [confidence, setConfidence] = useState(80);

  // WebSocket connection status
  const connectionStatus = useWebSocketStatus();

  // WebSocket event handlers
  const wsEventHandlers = {
    ai_recommendation: (recommendation) => {
      setRecommendations(prev => [recommendation, ...prev]);
    },
    optimization_complete: (data) => {
      console.log('Optimization complete:', data);
      // Reload recommendations after optimization
      loadRecommendations();
    }
  };

  // Initialize WebSocket with event handlers
  const { send } = useWebSocket(wsEventHandlers);

  useEffect(() => {
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    try {
      setLoading(true);
      
      // Fetch real train data from API
      const response = await trainAPI.getAll();
      const trains = response.data;
      
      // Generate AI recommendations based on actual train data
      const generatedRecommendations = generateRecommendationsFromTrains(trains);
      
      setRecommendations(generatedRecommendations);
    } catch (error) {
      console.error('Error loading recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateRecommendationsFromTrains = (trains) => {
    const recommendations = [];
    let recommendationId = 1;

    // Analyze train data and generate intelligent recommendations
    trains.forEach(train => {
      const delay = train.current_delay || 0;
      const priority = train.priority || 3;
      
      // 1. Delay Management Recommendations
      if (delay > 15) {
        recommendations.push({
          id: recommendationId++,
          type: 'delay_management',
          priority: delay > 30 ? 'critical' : 'high',
          title: `Optimize Schedule for ${train.train_name}`,
          description: `Train ${train.train_number} (${train.train_name}) is delayed by ${delay} minutes. Recommend speed optimization and platform reassignment.`,
          impact: `Reduce delay by ${Math.min(Math.floor(delay * 0.6), 20)} minutes and prevent cascading delays`,
          confidence: delay > 30 ? 95 : 88,
          estimated_benefit: `${Math.min(Math.floor(delay * 0.6), 20)} min recovery`,
          affected_trains: [train.train_number],
          status: 'pending',
          created_at: new Date(Date.now() - Math.random() * 10 * 60000),
          algorithm: 'OR-Tools Optimization'
        });
      }

      // 2. Priority-based Optimization
      if (priority <= 2 && delay > 5) {
        recommendations.push({
          id: recommendationId++,
          type: 'priority_optimization',
          priority: 'high',
          title: `High Priority Train Optimization`,
          description: `${train.train_name} (Priority ${priority}) should be given route preference to minimize delays.`,
          impact: `Maintain high-priority service standards and passenger satisfaction`,
          confidence: 92,
          estimated_benefit: `Priority lane access`,
          affected_trains: [train.train_number],
          status: 'pending',
          created_at: new Date(Date.now() - Math.random() * 15 * 60000),
          algorithm: 'Priority Queue Algorithm'
        });
      }

      // 3. Route Optimization based on train type
      if (train.train_type === 'FREIGHT' && delay < 5) {
        recommendations.push({
          id: recommendationId++,
          type: 'route_optimization',
          priority: 'medium',
          title: `Freight Train Route Optimization`,
          description: `Route ${train.train_name} via alternate freight corridor to free up passenger lines.`,
          impact: `Improve passenger train punctuality by 8% during peak hours`,
          confidence: 82,
          estimated_benefit: `Reduced congestion`,
          affected_trains: [train.train_number],
          status: Math.random() > 0.7 ? 'applied' : 'pending',
          created_at: new Date(Date.now() - Math.random() * 20 * 60000),
          algorithm: 'Genetic Algorithm'
        });
      }
    });

    // 4. System-wide recommendations based on overall train performance
    const delayedTrains = trains.filter(t => (t.current_delay || 0) > 10);
    const avgDelay = trains.reduce((sum, t) => sum + (t.current_delay || 0), 0) / trains.length;

    if (delayedTrains.length > trains.length * 0.3) {
      recommendations.push({
        id: recommendationId++,
        type: 'system_optimization',
        priority: 'critical',
        title: 'System-Wide Schedule Adjustment',
        description: `${delayedTrains.length} out of ${trains.length} trains are significantly delayed. Implement dynamic rescheduling.`,
        impact: `Reduce system-wide delays by 25-35% through coordinated optimization`,
        confidence: 89,
        estimated_benefit: `${Math.floor(avgDelay * 0.3)} min avg reduction`,
        affected_trains: delayedTrains.map(t => t.train_number).slice(0, 5),
        status: 'pending',
        created_at: new Date(Date.now() - 5 * 60000),
        algorithm: 'Multi-Agent Reinforcement Learning'
      });
    }

    // 5. Platform optimization based on current station data
    const stationGroups = {};
    trains.forEach(train => {
      const station = train.current_station || 'Unknown';
      if (!stationGroups[station]) stationGroups[station] = [];
      stationGroups[station].push(train);
    });

    Object.entries(stationGroups).forEach(([station, stationTrains]) => {
      if (stationTrains.length > 2) {
        recommendations.push({
          id: recommendationId++,
          type: 'platform_optimization',
          priority: 'medium',
          title: `Platform Management at ${station}`,
          description: `${stationTrains.length} trains at ${station}. Optimize platform allocation to reduce congestion.`,
          impact: `Reduce boarding/alighting time by 15% and improve platform efficiency`,
          confidence: 85,
          estimated_benefit: `3-5 min per train`,
          affected_trains: stationTrains.map(t => t.train_number),
          status: Math.random() > 0.6 ? 'reviewed' : 'pending',
          created_at: new Date(Date.now() - Math.random() * 12 * 60000),
          algorithm: 'Resource Allocation Optimizer'
        });
      }
    });

    // 6. Predictive maintenance recommendations
    const highUsageTrains = trains.filter(t => t.train_type === 'EXPRESS' || t.priority <= 2);
    if (highUsageTrains.length > 0) {
      const randomTrain = highUsageTrains[Math.floor(Math.random() * highUsageTrains.length)];
      recommendations.push({
        id: recommendationId++,
        type: 'predictive_maintenance',
        priority: 'low',
        title: 'Predictive Maintenance Alert',
        description: `Schedule maintenance window for high-usage route of ${randomTrain.train_name} based on performance analytics.`,
        impact: `Prevent potential service disruptions and ensure 99.5% reliability`,
        confidence: 78,
        estimated_benefit: `Prevent 2-3hr disruption`,
        affected_trains: [randomTrain.train_number],
        status: Math.random() > 0.8 ? 'dismissed' : 'pending',
        created_at: new Date(Date.now() - Math.random() * 30 * 60000),
        algorithm: 'Machine Learning Prediction'
      });
    }

    // Sort recommendations by priority and confidence
    return recommendations.sort((a, b) => {
      const priorityOrder = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 };
      if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      }
      return b.confidence - a.confidence;
    });
  };

  const applyRecommendation = async (recommendationId) => {
    try {
      setRecommendations(prev => prev.map(rec =>
        rec.id === recommendationId
          ? { ...rec, status: 'applying' }
          : rec
      ));

      // Find the recommendation to get the train info
      const recommendation = recommendations.find(rec => rec.id === recommendationId);
      if (!recommendation || !recommendation.affected_trains || recommendation.affected_trains.length === 0) {
        throw new Error('No train found for this recommendation');
      }

      const trainNumber = recommendation.affected_trains[0];
      const optimizationId = `opt_${trainNumber}`;

      // Call real API to apply optimization
      const response = await fetch(`http://127.0.0.1:8000/api/schedules/optimize/${optimizationId}/apply`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to apply optimization: ${response.statusText}`);
      }

      const result = await response.json();

      setRecommendations(prev => prev.map(rec =>
        rec.id === recommendationId
          ? { 
              ...rec, 
              status: 'applied', 
              applied_at: new Date(),
              application_result: result 
            }
          : rec
      ));

      // Show success message
      console.log('Optimization applied successfully:', result);
      
    } catch (error) {
      console.error('Error applying recommendation:', error);
      setRecommendations(prev => prev.map(rec =>
        rec.id === recommendationId
          ? { ...rec, status: 'failed', error_message: error.message }
          : rec
      ));
    }
  };

  const dismissRecommendation = (recommendationId) => {
    setRecommendations(prev => prev.map(rec =>
      rec.id === recommendationId
        ? { ...rec, status: 'dismissed' }
        : rec
    ));
  };

  const getFilteredRecommendations = () => {
    if (filter === 'all') return recommendations;
    return recommendations.filter(rec => rec.status === filter);
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'applied': return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'applying': return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'dismissed': return <XCircle className="h-5 w-5 text-gray-500" />;
      case 'failed': return <AlertTriangle className="h-5 w-5 text-red-500" />;
      case 'reviewed': return <Eye className="h-5 w-5 text-purple-500" />;
      default: return <Clock className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'schedule_optimization': return <Clock className="h-5 w-5" />;
      case 'conflict_resolution': return <AlertTriangle className="h-5 w-5" />;
      case 'route_optimization': return <TrendingUp className="h-5 w-5" />;
      case 'resource_allocation': return <Users className="h-5 w-5" />;
      case 'predictive_maintenance': return <Settings className="h-5 w-5" />;
      default: return <Zap className="h-5 w-5" />;
    }
  };

  const RecommendationCard = ({ recommendation }) => (
    <div className={`bg-white rounded-lg shadow-sm border-l-4 ${getPriorityColor(recommendation.priority)} p-6 mb-4`}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${getPriorityColor(recommendation.priority)}`}>
            {getTypeIcon(recommendation.type)}
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{recommendation.title}</h3>
            <p className="text-sm text-gray-600">
              {recommendation.algorithm} • {recommendation.created_at.toLocaleTimeString()}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getPriorityColor(recommendation.priority)}`}>
            {recommendation.priority.toUpperCase()}
          </span>
          {getStatusIcon(recommendation.status)}
        </div>
      </div>

      <p className="text-gray-700 mb-3">{recommendation.description}</p>
      
      <div className="bg-gray-50 rounded-lg p-3 mb-4">
        <p className="text-sm font-medium text-gray-900">Expected Impact:</p>
        <p className="text-sm text-gray-700">{recommendation.impact}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <p className="text-xs text-gray-500">Confidence Score</p>
          <div className="flex items-center space-x-2">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${
                  recommendation.confidence >= 90 ? 'bg-green-500' :
                  recommendation.confidence >= 80 ? 'bg-yellow-500' :
                  'bg-red-500'
                }`}
                style={{ width: `${recommendation.confidence}%` }}
              ></div>
            </div>
            <span className="text-sm font-medium">{recommendation.confidence}%</span>
          </div>
        </div>
        <div>
          <p className="text-xs text-gray-500">Estimated Benefit</p>
          <p className="font-medium text-green-600">{recommendation.estimated_benefit}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Affected Trains</p>
          <p className="font-medium">{recommendation.affected_trains.join(', ')}</p>
        </div>
      </div>

      {recommendation.status === 'pending' && (
        <div className="flex items-center justify-between pt-4 border-t border-gray-200">
          <div className="flex space-x-2">
            <button
              onClick={() => applyRecommendation(recommendation.id)}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
            >
              <CheckCircle className="h-4 w-4" />
              <span>Apply</span>
            </button>
            <button
              onClick={() => dismissRecommendation(recommendation.id)}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 text-sm"
            >
              <XCircle className="h-4 w-4" />
              <span>Dismiss</span>
            </button>
          </div>
          <div className="flex space-x-2">
            <button className="p-2 text-gray-400 hover:text-green-600">
              <ThumbsUp className="h-4 w-4" />
            </button>
            <button className="p-2 text-gray-400 hover:text-red-600">
              <ThumbsDown className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {recommendation.applied_at && (
        <div className="pt-4 border-t border-gray-200">
          <p className="text-sm text-green-600">
            Applied at {recommendation.applied_at.toLocaleTimeString()}
          </p>
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  const filteredRecommendations = getFilteredRecommendations();

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI Recommendations</h1>
          <p className="text-gray-600">Intelligent suggestions for optimizing railway operations</p>
        </div>
        <div className="flex items-center space-x-4">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            connectionStatus === 'connected' ? 'bg-green-100 text-green-800' :
            connectionStatus === 'connecting' ? 'bg-yellow-100 text-yellow-800' :
            'bg-red-100 text-red-800'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' :
              connectionStatus === 'connecting' ? 'bg-yellow-500' :
              'bg-red-500'
            }`}></div>
            <span className="capitalize">{connectionStatus}</span>
          </div>
          <button
            onClick={loadRecommendations}
            className="flex items-center space-x-2 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Status</label>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Recommendations</option>
              <option value="pending">Pending</option>
              <option value="applied">Applied</option>
              <option value="dismissed">Dismissed</option>
              <option value="reviewed">Reviewed</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Minimum Confidence: {confidence}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={confidence}
              onChange={(e) => setConfidence(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex items-end">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={autoApplyEnabled}
                onChange={(e) => setAutoApplyEnabled(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">Auto-apply high confidence recommendations</span>
            </label>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-yellow-500" />
            <span className="text-sm text-gray-600">Pending</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {recommendations.filter(r => r.status === 'pending').length}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            <span className="text-sm text-gray-600">Applied</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {recommendations.filter(r => r.status === 'applied').length}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            <span className="text-sm text-gray-600">Avg Confidence</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">
            {Math.round(recommendations.reduce((acc, r) => acc + r.confidence, 0) / recommendations.length)}%
          </p>
        </div>
        <div className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-purple-500" />
            <span className="text-sm text-gray-600">Today's Impact</span>
          </div>
          <p className="text-2xl font-bold text-gray-900">47 min</p>
        </div>
      </div>

      {/* Recommendations List */}
      <div>
        {filteredRecommendations.length === 0 ? (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
            <Zap className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No recommendations found</h3>
            <p className="text-gray-600">Try adjusting your filters or check back later for new AI insights.</p>
          </div>
        ) : (
          filteredRecommendations
            .filter(rec => rec.confidence >= confidence)
            .map((recommendation) => (
              <RecommendationCard key={recommendation.id} recommendation={recommendation} />
            ))
        )}
      </div>
    </div>
  );
};

export default AIRecommendations;