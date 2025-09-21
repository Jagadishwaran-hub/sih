import React, { useState, useEffect } from 'react';
import {
  Train,
  Clock,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  MapPin,
  Users,
  Zap,
  Calendar,
  ArrowUp,
  ArrowDown,
  Eye
} from 'lucide-react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';
import { trainAPI, scheduleAPI, kpiAPI } from '../services/api';

const OverviewDashboard = () => {
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('24h');
  const [metrics, setMetrics] = useState({
    totalTrains: 0,
    onTimePercentage: 85.4,
    avgDelay: 12.3,
    criticalAlerts: 3,
    sectionsManaged: 8,
    passengersSaved: 15420
  });

  const [recentActivity, setRecentActivity] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);
  const [delayDistribution, setDelayDistribution] = useState([]);
  const [stationPerformance, setStationPerformance] = useState([]);
  const [alertsData, setAlertsData] = useState([]);

  useEffect(() => {
    loadDashboardData();
  }, [timeRange]);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load multiple data sources in parallel
      const [
        metricsResponse,
        performanceResponse,
        delayResponse,
        alertsResponse
      ] = await Promise.all([
        kpiAPI.getMetrics({ timeRange }),
        kpiAPI.getPerformance('ALL', { timeRange }),
        kpiAPI.getDelayAnalysis({ timeRange }),
        trainAPI.getAll()
      ]);

      // Process metrics
      if (metricsResponse.data) {
        setMetrics(prev => ({ ...prev, ...metricsResponse.data }));
      }

      // Generate sample performance data
      setPerformanceData([
        { time: '00:00', onTime: 92, delayed: 8, cancelled: 0 },
        { time: '06:00', onTime: 88, delayed: 11, cancelled: 1 },
        { time: '12:00', onTime: 85, delayed: 13, cancelled: 2 },
        { time: '18:00', onTime: 82, delayed: 16, cancelled: 2 },
        { time: '24:00', onTime: 87, delayed: 12, cancelled: 1 },
      ]);

      // Generate delay distribution data
      setDelayDistribution([
        { range: '0-5 min', count: 156, fill: '#10B981' },
        { range: '5-15 min', count: 89, fill: '#F59E0B' },
        { range: '15-30 min', count: 45, fill: '#EF4444' },
        { range: '30+ min', count: 12, fill: '#8B5CF6' },
      ]);

      // Generate station performance data
      setStationPerformance([
        { station: 'NDLS', efficiency: 94, delays: 6, trains: 45 },
        { station: 'GZB', efficiency: 91, delays: 8, trains: 38 },
        { station: 'MB', efficiency: 88, delays: 12, trains: 42 },
        { station: 'BE', efficiency: 85, delays: 15, trains: 36 },
        { station: 'LKO', efficiency: 92, delays: 7, trains: 40 },
      ]);

      // Generate recent activity
      setRecentActivity([
        {
          id: 1,
          type: 'optimization',
          message: 'AI optimized schedule for 12951 Mumbai Rajdhani',
          time: '2 minutes ago',
          icon: Zap,
          color: 'text-blue-600'
        },
        {
          id: 2,
          type: 'alert',
          message: 'Platform 3 at NDLS approaching capacity limit',
          time: '5 minutes ago',
          icon: AlertTriangle,
          color: 'text-red-600'
        },
        {
          id: 3,
          type: 'arrival',
          message: '12553 Vaishali Express arrived on time at GZB',
          time: '8 minutes ago',
          icon: Train,
          color: 'text-green-600'
        },
        {
          id: 4,
          type: 'delay',
          message: '54373 Freight Special delayed by 25 minutes',
          time: '12 minutes ago',
          icon: Clock,
          color: 'text-yellow-600'
        },
        {
          id: 5,
          type: 'controller',
          message: 'Section Controller override applied for route MB-BE',
          time: '15 minutes ago',
          icon: Users,
          color: 'text-purple-600'
        }
      ]);

      setAlertsData([
        {
          id: 1,
          severity: 'high',
          title: 'Platform Congestion',
          description: 'Platform 2 at NDLS exceeding capacity',
          time: '5 min ago',
          status: 'active'
        },
        {
          id: 2,
          severity: 'medium',
          title: 'Schedule Conflict',
          description: 'Two trains scheduled at GZB within 3 minutes',
          time: '8 min ago',
          status: 'investigating'
        },
        {
          id: 3,
          severity: 'low',
          title: 'Weather Alert',
          description: 'Dense fog expected in Northern region',
          time: '12 min ago',
          status: 'monitoring'
        }
      ]);

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const StatCard = ({ title, value, change, icon: Icon, color = 'blue' }) => (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change && (
            <p className={`text-sm flex items-center mt-1 ${
              change > 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {change > 0 ? <ArrowUp className="h-4 w-4 mr-1" /> : <ArrowDown className="h-4 w-4 mr-1" />}
              {Math.abs(change)}% from yesterday
            </p>
          )}
        </div>
        <div className={`p-3 rounded-full bg-${color}-100`}>
          <Icon className={`h-6 w-6 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  const AlertCard = ({ alert }) => (
    <div className={`border-l-4 p-4 mb-3 bg-white rounded-r-lg shadow-sm ${
      alert.severity === 'high' ? 'border-red-500' :
      alert.severity === 'medium' ? 'border-yellow-500' :
      'border-blue-500'
    }`}>
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <h4 className="font-medium text-gray-900">{alert.title}</h4>
          <p className="text-sm text-gray-600 mt-1">{alert.description}</p>
          <div className="flex items-center mt-2 space-x-4">
            <span className="text-xs text-gray-400">{alert.time}</span>
            <span className={`text-xs px-2 py-1 rounded-full ${
              alert.status === 'active' ? 'bg-red-100 text-red-800' :
              alert.status === 'investigating' ? 'bg-yellow-100 text-yellow-800' :
              'bg-blue-100 text-blue-800'
            }`}>
              {alert.status}
            </span>
          </div>
        </div>
        <button className="text-gray-400 hover:text-gray-600">
          <Eye className="h-4 w-4" />
        </button>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Time Range Selector */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Railway Operations Overview</h1>
          <p className="text-gray-600">Real-time insights and performance metrics</p>
        </div>
        <div className="flex space-x-2">
          {['1h', '24h', '7d', '30d'].map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-2 text-sm rounded-md transition-colors ${
                timeRange === range
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-600 hover:bg-gray-50 border border-gray-300'
              }`}
            >
              {range.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-6">
        <StatCard
          title="Active Trains"
          value={metrics.totalTrains}
          change={2.3}
          icon={Train}
          color="blue"
        />
        <StatCard
          title="On-Time Performance"
          value={`${metrics.onTimePercentage}%`}
          change={-1.2}
          icon={Clock}
          color="green"
        />
        <StatCard
          title="Avg Delay (min)"
          value={metrics.avgDelay}
          change={-0.8}
          icon={TrendingDown}
          color="yellow"
        />
        <StatCard
          title="Critical Alerts"
          value={metrics.criticalAlerts}
          change={-2}
          icon={AlertTriangle}
          color="red"
        />
        <StatCard
          title="Sections Managed"
          value={metrics.sectionsManaged}
          icon={MapPin}
          color="purple"
        />
        <StatCard
          title="AI Optimizations"
          value="47"
          change={12.5}
          icon={Zap}
          color="indigo"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Trend */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Performance Trend</h3>
            <div className="flex space-x-4 text-sm">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                <span>On Time</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                <span>Delayed</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                <span>Cancelled</span>
              </div>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="onTime" stackId="1" stroke="#10B981" fill="#10B981" />
              <Area type="monotone" dataKey="delayed" stackId="1" stroke="#F59E0B" fill="#F59E0B" />
              <Area type="monotone" dataKey="cancelled" stackId="1" stroke="#EF4444" fill="#EF4444" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Delay Distribution */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Delay Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={delayDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={5}
                dataKey="count"
              >
                {delayDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="grid grid-cols-2 gap-2 mt-4">
            {delayDistribution.map((item, index) => (
              <div key={index} className="flex items-center text-sm">
                <div 
                  className="w-3 h-3 rounded-full mr-2"
                  style={{ backgroundColor: item.fill }}
                ></div>
                <span>{item.range}: {item.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Station Performance */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Station Performance</h3>
          <div className="space-y-4">
            {stationPerformance.map((station, index) => (
              <div key={index} className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{station.station}</p>
                  <p className="text-sm text-gray-500">{station.trains} trains</p>
                </div>
                <div className="text-right">
                  <p className={`font-medium ${
                    station.efficiency >= 90 ? 'text-green-600' :
                    station.efficiency >= 85 ? 'text-yellow-600' :
                    'text-red-600'
                  }`}>
                    {station.efficiency}%
                  </p>
                  <p className="text-sm text-gray-500">{station.delays} delays</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-4">
            {recentActivity.map((activity) => {
              const Icon = activity.icon;
              return (
                <div key={activity.id} className="flex items-start space-x-3">
                  <div className={`p-2 rounded-full bg-gray-100`}>
                    <Icon className={`h-4 w-4 ${activity.color}`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">{activity.message}</p>
                    <p className="text-xs text-gray-500">{activity.time}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Active Alerts */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Alerts</h3>
          <div className="space-y-2">
            {alertsData.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default OverviewDashboard;