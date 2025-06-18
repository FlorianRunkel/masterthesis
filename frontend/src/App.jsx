import React, { useState, useEffect } from 'react';
import { ThemeProvider, useTheme } from '@mui/material/styles';
import { CssBaseline, Box, useMediaQuery } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/sidebar/Sidebar';
import LinkedInInput from './pages/linkedInInput';
import BatchUpload from './pages/batchUpload';
import ManualInput from './pages/manualInput';
import CandidatesPage from './pages/candidates';
import Login from './pages/login';
import SettingsPage from './pages/settings';

const App = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    setIsLoggedIn(localStorage.getItem('isLoggedIn') === 'true');
  }, []);

  const handleLogin = () => {
    setIsLoggedIn(true);
  };
  const handleLogout = () => {
    localStorage.clear();
    setIsLoggedIn(false);
  };

  if (!isLoggedIn) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Sidebar onLogout={handleLogout} />
        <Box 
          component="main"
          sx={{
            marginLeft: isMobile ? 0 : '260px',
            marginTop: isMobile ? '70px' : 0,
            padding: { xs: '8px', sm: '16px', md: '32px', lg: '40px', xl: '56px' },
            minHeight: '100vh',
            width: '100%',
            maxWidth: { xs: '100vw', sm: 600, md: 1200, lg: 1200, xl: 1536 },
            mx: 'auto',
            boxSizing: 'border-box',
            bgcolor: 'background.default'
          }}
        >
          <Routes>
            <Route path="/" element={<ManualInput />} />
            <Route path="/batch" element={<BatchUpload />} />
            <Route path="/linkedin" element={<LinkedInInput />} />
            <Route path="/candidates" element={<CandidatesPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App; 