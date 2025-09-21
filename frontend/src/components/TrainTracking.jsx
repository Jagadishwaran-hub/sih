import React, { useState, useEffect } from 'react';
import {
  Train,
  MapPin,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Pause,
  Play,
  Filter,
  Search,
  RefreshCw,
  Navigation,
  Zap,
  Info
} from 'lucide-react';
import { trainAPI, wsManager, formatDateTime, calculateDelay, getStatusColor, getTrainTypeIcon } from '../services/api';

const TrainTracking = () => {
  const [trains, setTrains] = useState([]);
  const [filteredTrains, setFilteredTrains] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTrain, setSelectedTrain] = useState(null);
  const [filters, setFilters] = useState({
    status: 'all',
    trainType: 'all',
    priority: 'all'
  });
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    loadTrains();
    
    // Set up WebSocket listeners for real-time updates
    wsManager.on('train_update', handleTrainUpdate);
    wsManager.on('train_position', handlePositionUpdate);
    
    // Auto refresh every 30 seconds if enabled
    const interval = autoRefresh ? setInterval(loadTrains, 30000) : null;
    
    return () => {
      if (interval) clearInterval(interval);
      wsManager.off('train_update', handleTrainUpdate);
      wsManager.off('train_position', handlePositionUpdate);
    };
  }, [autoRefresh]);

  useEffect(() => {
    filterTrains();
  }, [trains, searchQuery, filters]);

  const loadTrains = async () => {
    try {
      setLoading(true);
      const response = await trainAPI.getAll();
      setTrains(response.data || []);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading trains:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTrainUpdate = (update) => {
    setTrains(prev => prev.map(train => 
      train.train_id === update.train_id 
        ? { ...train, ...update }
        : train
    ));
    setLastUpdated(new Date());
  };

  const handlePositionUpdate = (update) => {
    setTrains(prev => prev.map(train => 
      train.train_id === update.train_id 
        ? { ...train, current_location: update.location, last_updated: update.timestamp }
        : train
    ));
  };

  const filterTrains = () => {
    let filtered = trains;

    // Search filter
    if (searchQuery) {
      filtered = filtered.filter(train =>
        train.train_number?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        train.train_name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        train.current_location?.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(train => train.status === filters.status);
    }

    // Train type filter
    if (filters.trainType !== 'all') {
      filtered = filtered.filter(train => train.train_type === filters.trainType);
    }

    // Priority filter
    if (filters.priority !== 'all') {
      filtered = filtered.filter(train => train.priority.toString() === filters.priority);
    }

    setFilteredTrains(filtered);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ON_TIME':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'DELAYED':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'CRITICAL_DELAY':
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      case 'CANCELLED':
        return <XCircle className="h-5 w-5 text-gray-500" />;
      case 'DEPARTED':
        return <Navigation className="h-5 w-5 text-blue-500" />;
      case 'ARRIVED':
        return <MapPin className="h-5 w-5 text-purple-500" />;
      default:
        return <Info className="h-5 w-5 text-gray-400" />;
    }
  };

  const getPriorityBadge = (priority) => {
    const colors = {
      1: 'bg-red-100 text-red-800',
      2: 'bg-orange-100 text-orange-800',
      3: 'bg-yellow-100 text-yellow-800',
      4: 'bg-blue-100 text-blue-800',
      5: 'bg-gray-100 text-gray-800'
    };
    
    return (
      <span className={`px-2 py-1 text-xs font-medium rounded-full ${colors[priority] || colors[5]}`}>
        P{priority}
      </span>
    );
  };

  const TrainCard = ({ train }) => {
    const delay = train.scheduled_arrival && train.estimated_arrival 
      ? calculateDelay(train.scheduled_arrival, train.estimated_arrival)
      : 0;

    return (
      <div 
        className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 hover:shadow-md transition-shadow cursor-pointer ${
          selectedTrain?.train_id === train.train_id ? 'ring-2 ring-blue-500' : ''
        }`}
        onClick={() => setSelectedTrain(train)}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">{getTrainTypeIcon(train.train_type)}</div>
            <div>
              <h3 className="font-semibold text-gray-900">{train.train_number}</h3>
              <p className="text-sm text-gray-600">{train.train_name}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {getPriorityBadge(train.priority)}
            {getStatusIcon(train.status)}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-3">
          <div>
            <p className="text-xs text-gray-500">Current Location</p>
            <p className="font-medium text-gray-900">{train.current_location || 'Unknown'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Next Station</p>
            <p className="font-medium text-gray-900">{train.next_station || 'N/A'}</p>
          </div>
        </div>

        {train.scheduled_arrival && (
          <div className="grid grid-cols-2 gap-4 mb-3">
            <div>
              <p className="text-xs text-gray-500">Scheduled Arrival</p>
              <p className="font-medium text-gray-900">
                {formatDateTime(train.scheduled_arrival)}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Estimated Arrival</p>
              <p className={`font-medium ${delay > 0 ? 'text-red-600' : 'text-green-600'}`}>
                {formatDateTime(train.estimated_arrival)}
                {delay > 0 && (
                  <span className="text-xs ml-1">(+{delay}m)</span>
                )}
              </p>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between">
          <span className={`text-sm font-medium ${getStatusColor(train.status)}`}>
            {train.status?.replace('_', ' ')}
          </span>
          <span className="text-xs text-gray-500">
            Updated {new Date(train.last_updated || Date.now()).toLocaleTimeString()}
          </span>
        </div>
      </div>
    );
  };

  const TrainDetails = ({ train }) => {
    if (!train) return null;

    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4">
            <div className="text-3xl">{getTrainTypeIcon(train.train_type)}</div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">{train.train_number}</h2>
              <p className="text-gray-600">{train.train_name}</p>
              <div className="flex items-center space-x-2 mt-1">
                {getPriorityBadge(train.priority)}
                <span className="text-sm text-gray-500">{train.train_type}</span>
              </div>
            </div>
          </div>
          <div className="text-right">
            {getStatusIcon(train.status)}
            <p className={`text-sm font-medium mt-1 ${getStatusColor(train.status)}`}>
              {train.status?.replace('_', ' ')}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Current Status</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Current Location</span>
                <span className="font-medium">{train.current_location || 'Unknown'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Next Station</span>
                <span className="font-medium">{train.next_station || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Speed</span>
                <span className="font-medium">{train.current_speed || 0} km/h</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Direction</span>
                <span className="font-medium">{train.direction || 'N/A'}</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Schedule Information</h3>
            <div className="space-y-3">
              {train.scheduled_arrival && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Scheduled Arrival</span>
                  <span className="font-medium">{formatDateTime(train.scheduled_arrival)}</span>
                </div>
              )}
              {train.estimated_arrival && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Estimated Arrival</span>
                  <span className="font-medium">{formatDateTime(train.estimated_arrival)}</span>
                </div>
              )}
              {train.scheduled_departure && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Scheduled Departure</span>
                  <span className="font-medium">{formatDateTime(train.scheduled_departure)}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-600">Current Delay</span>
                <span className={`font-medium ${train.current_delay > 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {train.current_delay || 0} minutes
                </span>
              </div>
            </div>
          </div>
        </div>

        {train.route && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Route</h3>
            <div className="flex items-center space-x-2 flex-wrap">
              {train.route.map((station, index) => (
                <React.Fragment key={station}>
                  <span className={`px-3 py-1 rounded-full text-sm ${
                    station === train.current_location 
                      ? 'bg-blue-100 text-blue-800 font-medium' 
                      : 'bg-gray-100 text-gray-700'
                  }`}>
                    {station}
                  </span>
                  {index < train.route.length - 1 && (
                    <span className="text-gray-400">→</span>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        )}

        {train.alerts && train.alerts.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Active Alerts</h3>
            <div className="space-y-2">
              {train.alerts.map((alert, index) => (
                <div key={index} className="flex items-center space-x-2 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  <span className="text-sm text-yellow-800">{alert.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Train Tracking</h1>
          <p className="text-gray-600">Real-time monitoring of train positions and status</p>
        </div>
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm ${
              autoRefresh 
                ? 'bg-green-100 text-green-700' 
                : 'bg-gray-100 text-gray-700'
            }`}
          >
            {autoRefresh ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            <span>{autoRefresh ? 'Auto Refresh On' : 'Auto Refresh Off'}</span>
          </button>
          <button
            onClick={loadTrains}
            className="flex items-center space-x-2 px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search trains..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <select
            value={filters.status}
            onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="ON_TIME">On Time</option>
            <option value="DELAYED">Delayed</option>
            <option value="CRITICAL_DELAY">Critical Delay</option>
            <option value="CANCELLED">Cancelled</option>
          </select>

          <select
            value={filters.trainType}
            onChange={(e) => setFilters(prev => ({ ...prev, trainType: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Types</option>
            <option value="EXPRESS">Express</option>
            <option value="PASSENGER">Passenger</option>
            <option value="FREIGHT">Freight</option>
          </select>

          <select
            value={filters.priority}
            onChange={(e) => setFilters(prev => ({ ...prev, priority: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Priorities</option>
            <option value="1">Priority 1</option>
            <option value="2">Priority 2</option>
            <option value="3">Priority 3</option>
            <option value="4">Priority 4</option>
            <option value="5">Priority 5</option>
          </select>
        </div>
        
        <div className="mt-4 flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Showing {filteredTrains.length} of {trains.length} trains
          </p>
          <p className="text-sm text-gray-500">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Train List */}
        <div className="lg:col-span-2 space-y-4">
          {filteredTrains.length === 0 ? (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
              <Train className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No trains found</h3>
              <p className="text-gray-600">Try adjusting your search criteria or filters.</p>
            </div>
          ) : (
            filteredTrains.map((train) => (
              <TrainCard key={train.train_id} train={train} />
            ))
          )}
        </div>

        {/* Train Details Panel */}
        <div className="lg:col-span-1">
          {selectedTrain ? (
            <TrainDetails train={selectedTrain} />
          ) : (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
              <Info className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Select a train</h3>
              <p className="text-gray-600">Click on a train card to view detailed information.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainTracking;