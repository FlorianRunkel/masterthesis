import React, { useState, useEffect } from 'react';
import { ThemeProvider, useTheme } from '@mui/material/styles';
import { CssBaseline, Box, useMediaQuery } from '@mui/material';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Sidebar from './components/sidebar/Sidebar';
import LinkedInInput from './pages/linkedInInput';
import BatchUpload from './pages/batchUpload';
import ManualInput from './pages/manualInput';
import CandidatesPage from './pages/candidates';
import Login from './pages/login';
import SettingsPage from './pages/settings';
import Index from './pages/Index';
import FeedbackPage from './pages/feedback';

const AppContent = ({ onLogout, onLogin }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  const navigate = useNavigate();

  useEffect(() => {
    if (localStorage.getItem('isLoggedIn') === 'true') {
      const user = JSON.parse(localStorage.getItem('user'));
      if (window.location.pathname === '/login') {
        navigate('/');
      }
      if (window.location.pathname === '/settings' && !user?.admin) {
        navigate('/');
      }
    }
  }, [navigate]);

  const handleLogoutAndRedirect = () => {
    localStorage.clear();
    onLogout();
    navigate('/login');
  };

  const sidebarWidth = sidebarCollapsed ? 100 : 280;

  return (
    <Box sx={{ minHeight: '100vh' }}>
      <Sidebar 
        onLogout={handleLogoutAndRedirect} 
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <Box 
        component="main"
        sx={{
          flexGrow: 1,
          marginLeft: { xs: 0, md: `${sidebarWidth}px` },
          marginTop: isMobile ? '70px' : 0,
          padding: { xs: 2, sm: 3, md: 4 },
          transition: 'margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          boxSizing: 'border-box',
          bgcolor: 'background.default',
          width: { xs: '100%', md: `calc(100% - ${sidebarWidth}px)` }
        }}
      >
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/manual" element={<ManualInput />} />
          <Route path="/batch" element={<BatchUpload />} />
          <Route path="/linkedin" element={<LinkedInInput />} />
          <Route path="/candidates" element={<CandidatesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/feedback" element={<FeedbackPage />} />
        </Routes>
      </Box>
    </Box>
  );
};

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    setIsLoggedIn(localStorage.getItem('isLoggedIn') === 'true');
  }, []);

  const handleLogout = () => {
    setIsLoggedIn(false);
  };

  return (
    <ThemeProvider theme={useTheme()}>
      <CssBaseline />
      <Router>
        {isLoggedIn ? (
          <AppContent onLogout={handleLogout}/>
        ) : (
          <Routes>
            <Route path="*" element={<Login onLogin={() => setIsLoggedIn(true)} />} />
          </Routes>
        )}
      </Router>
    </ThemeProvider>
  );
};

export default App; 