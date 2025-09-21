import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access 
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Train API
export const trainAPI = {
  getAll: () => api.get('/api/trains'),
  getById: (id) => api.get(`/api/trains/${id}`),
  create: (train) => api.post('/api/trains', train),
  update: (id, train) => api.put(`/api/trains/${id}`, train),
  delete: (id) => api.delete(`/api/trains/${id}`),
  getStatus: (id) => api.get(`/api/trains/${id}/status`),
  updatePosition: (id, position) => api.patch(`/api/trains/${id}/position`, position),
};

// Schedule API
export const scheduleAPI = {
  getAll: () => api.get('/api/schedules'),
  getByStation: (stationCode) => api.get(`/api/schedules/station/${stationCode}`),
  getByTrain: (trainId) => api.get(`/api/schedules/train/${trainId}`),
  create: (schedule) => api.post('/api/schedules', schedule),
  update: (id, schedule) => api.put(`/api/schedules/${id}`, schedule),
  delete: (id) => api.delete(`/api/schedules/${id}`),
  optimize: (params) => api.post('/api/schedules/optimize', params),
  resolveConflict: (conflictId) => api.post(`/api/schedules/conflicts/${conflictId}/resolve`),
}; 

// Simulation API
export const simulationAPI = {
  getScenarios: () => api.get('/api/simulation/scenarios'),
  getScenario: (scenarioId) => api.get(`/api/simulation/scenarios/${scenarioId}`),
  runSimulation: (scenarioId, params) => api.post(`/api/simulation/run/${scenarioId}`, params),
  getResults: (simulationId) => api.get(`/api/simulation/results/${simulationId}`),
  createCustomScenario: (scenario) => api.post('/api/simulation/scenarios/custom', scenario),
  validateScenario: (scenario) => api.post('/api/simulation/scenarios/validate', scenario),
};

// KPI API
export const kpiAPI = {
  getMetrics: (timeRange) => api.get('/api/kpis/metrics', { params: timeRange }),
  getPerformance: (stationCode, timeRange) => 
    api.get(`/api/kpis/performance/${stationCode}`, { params: timeRange }),
  getDelayAnalysis: (timeRange) => api.get('/api/kpis/delays', { params: timeRange }),
  getAuditLog: (filters) => api.get('/api/kpis/audit', { params: filters }),
  exportMetrics: (format, timeRange) => 
    api.get('/api/kpis/export', { params: { format, ...timeRange }, responseType: 'blob' }),
};

// WebSocket connection management
class WebSocketManager {
  constructor() {
    this.socket = null;
    this.listeners = new Map();
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second
    this.heartbeatInterval = null;
  }

  connect() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      return; // Already connected
    }
    
    const wsUrl = (API_BASE_URL.replace('http', 'ws')) + '/ws';
    
    console.log('Connecting to WebSocket:', wsUrl);
    this.emit('connecting');
    
    try {
      console.log('Connecting to WebSocket:', wsUrl);
      this.socket = new WebSocket(wsUrl);
      
      this.socket.onopen = () => {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000; // Reset delay
        this.emit('connected');
        this.startHeartbeat();
        
        // Subscribe to events
        this.send('subscribe', ['train_update', 'system_stats', 'ai_recommendation', 'alert']);
      };
      
      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data);
          
          // Handle pong responses specifically
          if (data.type === 'pong') {
            console.log('Received pong from server');
            return; // Don't emit pong events
          }
          
          this.emit(data.type, data.payload || data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      this.socket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        this.stopHeartbeat();
        this.emit('disconnected');
        
        if (!event.wasClean) {
          this.attemptReconnect();
        }
      };
      
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
      
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      this.attemptReconnect();
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close(1000, 'Intentional disconnect');
      this.socket = null;
      this.isConnected = false;
      this.stopHeartbeat();
    }
  }

  startHeartbeat() {
    this.stopHeartbeat(); // Clear any existing interval
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.socket && this.socket.readyState === WebSocket.OPEN) {
        console.log('Sending ping to server');
        this.send('ping', { timestamp: new Date().toISOString() });
      }
    }, 30000); // Ping every 30 seconds
  }

  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      
      console.log(`Attempting to reconnect in ${this.reconnectDelay/1000}s (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.emit('reconnecting', { attempt: this.reconnectAttempts, maxAttempts: this.maxReconnectAttempts });
      
      setTimeout(() => {
        if (!this.isConnected) { // Only reconnect if still disconnected
          this.connect();
        }
      }, this.reconnectDelay);
      
      // Exponential backoff with jitter
      this.reconnectDelay = Math.min(this.reconnectDelay * 2 + Math.random() * 1000, 30000);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('max_reconnect_attempts');
    }
  }

  send(type, payload) {
    if (this.isConnected && this.socket && this.socket.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ type, payload });
      this.socket.send(message);
    } else {
      console.warn('WebSocket not connected. Message not sent:', { type, payload });
    }
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebSocket event handler for ${event}:`, error);
        }
      });
    }
  }
}

// Create singleton WebSocket manager
export const wsManager = new WebSocketManager();

// Utility functions
export const formatDateTime = (date) => {
  return new Intl.DateTimeFormat('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Asia/Kolkata'
  }).format(new Date(date));
};

export const calculateDelay = (scheduledTime, actualTime) => {
  const scheduled = new Date(scheduledTime);
  const actual = new Date(actualTime);
  const diffInMinutes = Math.floor((actual - scheduled) / (1000 * 60));
  return diffInMinutes;
};

export const getStatusColor = (status) => {
  const statusColors = {
    'ON_TIME': 'text-green-500',
    'DELAYED': 'text-yellow-500',
    'CRITICAL_DELAY': 'text-red-500',
    'CANCELLED': 'text-gray-500',
    'DEPARTED': 'text-blue-500',
    'ARRIVED': 'text-purple-500',
  };
  return statusColors[status] || 'text-gray-400';
};

export const getTrainTypeIcon = (trainType) => {
  const typeIcons = {
    'EXPRESS': '🚄',
    'PASSENGER': '🚞',
    'FREIGHT': '🚛',
    'SUPERFAST': '⚡',
    'LOCAL': '🚃',
  };
  return typeIcons[trainType] || '🚂';
};

export default api;