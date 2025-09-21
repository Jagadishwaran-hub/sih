import React, { useState } from 'react';
import DashboardLayout from './components/DashboardLayout';
import OverviewDashboard from './components/OverviewDashboard';
import TrainTracking from './components/TrainTracking';
import AIRecommendations from './components/AIRecommendations';
// Import other components as they are created
// import ScheduleManagement from './components/ScheduleManagement';
// import Simulation from './components/Simulation';
// import NetworkMap from './components/NetworkMap';
// import Analytics from './components/Analytics';
// import ControllerPanel from './components/ControllerPanel';

import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('overview');

  const renderContent = () => {
    switch (currentView) {
      case 'overview':
        return <OverviewDashboard />;
      case 'tracking':
        return <TrainTracking />;
      case 'schedule':
        return (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Schedule Management</h2>
            <p className="text-gray-600">Schedule management component coming soon...</p>
          </div>
        );
      case 'ai':
        return <AIRecommendations />;
      case 'simulation':
        return (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Simulation</h2>
            <p className="text-gray-600">Simulation component coming soon...</p>
          </div>
        );
      case 'map':
        return (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Network Map</h2>
            <p className="text-gray-600">Network map component coming soon...</p>
          </div>
        );
      case 'analytics':
        return (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">KPIs & Analytics</h2>
            <p className="text-gray-600">Analytics component coming soon...</p>
          </div>
        );
      case 'control':
        return (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Controller Panel</h2>
            <p className="text-gray-600">Controller panel component coming soon...</p>
          </div>
        );
      default:
        return <OverviewDashboard />;
    }
  };

  return (
    <DashboardLayout currentView={currentView} onViewChange={setCurrentView}>
      {renderContent()}
    </DashboardLayout>
  );
}

export default App;
