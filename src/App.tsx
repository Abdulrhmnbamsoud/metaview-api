// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';

// 4 Primary destinations
import { ExecutiveOverview } from './pages/ExecutiveOverview';
import { UnifiedExplore } from './pages/UnifiedExplore';
import { DetectionRules } from './pages/DetectionRules';
import { SourcesAndSystem } from './pages/SourcesAndSystem';

// Backward-compatible views & tools
import { IncidentRegistry } from './pages/IncidentRegistry';
import { SaudiFraud } from './pages/SaudiFraud';
import { FraudPatterns } from './pages/FraudPatterns';
import { Dashboard } from './pages/Dashboard';
import { WorldMap } from './pages/WorldMap';
import { Repository } from './pages/Repository';
import { AiLayer } from './pages/AiLayer';
import { SystemHealth } from './pages/SystemHealth';
import { Operations } from './pages/Operations';
import { Sources } from './pages/Sources';

export const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          {/* 4 Main Destinations */}
          <Route path="/" element={<ExecutiveOverview />} />
          <Route path="/explore" element={<UnifiedExplore />} />
          <Route path="/rules" element={<DetectionRules />} />
          <Route path="/system" element={<SourcesAndSystem />} />

          {/* Deep links & Backward-compatible aliases */}
          <Route path="/explorer" element={<UnifiedExplore />} />
          <Route path="/incident-registry" element={<IncidentRegistry />} />
          <Route path="/saudi-fraud" element={<SaudiFraud />} />
          <Route path="/patterns" element={<FraudPatterns />} />
          <Route path="/sources" element={<SourcesAndSystem />} />
          <Route path="/operations" element={<SourcesAndSystem />} />
          <Route path="/system-health" element={<SystemHealth />} />
          <Route path="/global" element={<Dashboard />} />
          <Route path="/map" element={<WorldMap />} />
          <Route path="/repository" element={<Repository />} />
          <Route path="/ai-layer" element={<AiLayer />} />

          {/* Catch-all redirect */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;
