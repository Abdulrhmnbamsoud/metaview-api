// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Explorer } from './pages/Explorer';
import { WorldMap } from './pages/WorldMap';
import { Operations } from './pages/Operations';
import { Repository } from './pages/Repository';
import { Sources } from './pages/Sources';
import { AiLayer } from './pages/AiLayer';
import { SaudiFraud } from './pages/SaudiFraud';
import { IncidentRegistry } from './pages/IncidentRegistry';

export const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/map" element={<WorldMap />} />
          <Route path="/operations" element={<Operations />} />
          <Route path="/repository" element={<Repository />} />
          <Route path="/sources" element={<Sources />} />
          <Route path="/ai-layer" element={<AiLayer />} />
          <Route path="/saudi-fraud" element={<SaudiFraud />} />
          <Route path="/incident-registry" element={<IncidentRegistry />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;
