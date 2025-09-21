import React, { useState, useEffect } from 'react';
import { 
  Train, 
  Clock, 
  AlertTriangle, 
  Activity, 
  Users, 
  Settings,
  Bell,
  Search,
  Menu,
  X,
  MapPin,
  BarChart3,
  Zap
} from 'lucide-react';
import { useWebSocket, useWebSocketStatus } from '../hooks/useWebSocket';

const DashboardLayout = ({ children, currentView, onViewChange }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [notifications, setNotifications] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [systemStats, setSystemStats] = useState({
    totalTrains: 0,
    onTimePercentage: 0,
    activeAlerts: 0,
    avgDelay: 0
  });

  // WebSocket connection status
  const connectionStatus = useWebSocketStatus();

  // WebSocket event handlers
  const wsEventHandlers = {
    train_update: (data) => {
      console.log('Train update:', data);
    },
    alert: (data) => {
      setNotifications(prev => [...prev, {
        id: Date.now(),
        type: data.severity || 'info',
        title: data.title,
        message: data.message,
        timestamp: new Date()
      }]);
    },
    system_stats: (stats) => {
      setSystemStats(stats);
    }
  };

  // Initialize WebSocket with event handlers
  const { connect, disconnect, send } = useWebSocket(wsEventHandlers);

  // Clean up notifications
  const dismissNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const navigation = [
    {
      name: 'Overview',
      id: 'overview',
      icon: Activity,
      description: 'System overview and key metrics'
    },
    {
      name: 'Train Tracking',
      id: 'tracking',
      icon: Train,
      description: 'Real-time train positions and status'
    },
    {
      name: 'Schedule Management',
      id: 'schedule',
      icon: Clock,
      description: 'Train schedules and conflicts'
    },
    {
      name: 'AI Recommendations',
      id: 'ai',
      icon: Zap,
      description: 'AI-powered optimization suggestions'
    },
    {
      name: 'Simulation',
      id: 'simulation',
      icon: BarChart3,
      description: 'What-if scenario testing'
    },
    {
      name: 'Network Map',
      id: 'map',
      icon: MapPin,
      description: 'Interactive railway network view'
    },
    {
      name: 'KPIs & Analytics',
      id: 'analytics',
      icon: BarChart3,
      description: 'Performance metrics and reports'
    },
    {
      name: 'Controller Panel',
      id: 'control',
      icon: Users,
      description: 'Section controller tools'
    }
  ];

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500';
      case 'connecting': return 'text-yellow-500';
      case 'disconnected': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  return (
    <div className="w-full h-screen bg-gray-50 flex overflow-hidden min-w-full">
      {/* Sidebar */}
      <div className={`bg-white shadow-lg transition-all duration-300 ${
        sidebarOpen ? 'w-64' : 'w-16'
      } flex flex-col flex-shrink-0`}>
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200">
          {sidebarOpen && (
            <div className="flex items-center space-x-2">
              <Train className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-lg font-bold text-gray-900">Railway AI</h1>
                <p className="text-xs text-gray-500">Control Center</p>
              </div>
            </div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100"
          >
            {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>

        {/* System Status */}
        {sidebarOpen && (
          <div className="px-4 py-3 border-b border-gray-200">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">System Status</span>
              <span className={`font-medium ${getConnectionStatusColor()}`}>
                {getConnectionStatusText()}
              </span>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div className="text-center p-2 bg-gray-50 rounded">
                <div className="font-semibold text-gray-900">{systemStats.totalTrains}</div>
                <div className="text-gray-500">Active Trains</div>
              </div>
              <div className="text-center p-2 bg-gray-50 rounded">
                <div className="font-semibold text-green-600">{systemStats.onTimePercentage}%</div>
                <div className="text-gray-500">On Time</div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = currentView === item.id;
            const Icon = item.icon;
            
            return (
              <button
                key={item.id}
                onClick={() => onViewChange(item.id)}
                className={`w-full flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors ${
                  isActive
                    ? 'bg-blue-100 text-blue-700 border-r-2 border-blue-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
                title={sidebarOpen ? '' : item.name}
              >
                <Icon className={`h-5 w-5 ${sidebarOpen ? 'mr-3' : 'mx-auto'}`} />
                {sidebarOpen && (
                  <div className="flex-1 text-left">
                    <div>{item.name}</div>
                    <div className="text-xs text-gray-500 mt-0.5">{item.description}</div>
                  </div>
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        {sidebarOpen && (
          <div className="p-4 border-t border-gray-200">
            <button className="w-full flex items-center px-3 py-2 text-sm text-gray-600 hover:bg-gray-50 rounded-md">
              <Settings className="h-4 w-4 mr-3" />
              Settings
            </button>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h2 className="text-xl font-semibold text-gray-900">
                {navigation.find(nav => nav.id === currentView)?.name || 'Dashboard'}
              </h2>
              
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search trains, stations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Alerts */}
              {systemStats.activeAlerts > 0 && (
                <div className="flex items-center space-x-2 px-3 py-1 bg-red-50 text-red-700 rounded-full text-sm">
                  <AlertTriangle className="h-4 w-4" />
                  <span>{systemStats.activeAlerts} Alerts</span>
                </div>
              )}

              {/* Notifications */}
              <div className="relative">
                <button className="p-2 text-gray-400 hover:text-gray-600 relative">
                  <Bell className="h-5 w-5" />
                  {notifications.length > 0 && (
                    <span className="absolute -top-1 -right-1 h-4 w-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {notifications.length}
                    </span>
                  )}
                </button>
              </div>

              {/* User Menu */}
              <div className="flex items-center space-x-2">
                <div className="h-8 w-8 bg-blue-600 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-medium">SC</span>
                </div>
                <span className="text-sm font-medium text-gray-700">Section Controller</span>
              </div>
            </div>
          </div>
        </header>

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto bg-gray-50 p-6 w-full">
          <div className="w-full max-w-none">
            {children}
          </div>
        </main>
      </div>

      {/* Notifications Panel */}
      {notifications.length > 0 && (
        <div className="fixed top-20 right-6 w-80 space-y-2 z-50">
          {notifications.slice(0, 5).map((notification) => (
            <div
              key={notification.id}
              className={`p-4 rounded-lg shadow-lg border-l-4 bg-white ${
                notification.type === 'error' ? 'border-red-500' :
                notification.type === 'warning' ? 'border-yellow-500' :
                notification.type === 'success' ? 'border-green-500' :
                'border-blue-500'
              }`}
            >
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900">{notification.title}</h4>
                  <p className="text-sm text-gray-600 mt-1">{notification.message}</p>
                  <p className="text-xs text-gray-400 mt-2">
                    {notification.timestamp.toLocaleTimeString()}
                  </p>
                </div>
                <button
                  onClick={() => dismissNotification(notification.id)}
                  className="ml-2 text-gray-400 hover:text-gray-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DashboardLayout;