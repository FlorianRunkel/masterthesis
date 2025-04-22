import React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import theme from './theme';
import Sidebar from './components/layout/Sidebar';
import LinkedInInput from './components/candidates/LinkedInInput';
import BatchUpload from './components/candidates/BatchUpload';
import ManualInput from './components/candidates/ManualInput';
import CandidatesPage from './pages/CandidatesPage';

const App = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex' }}>
          <Sidebar />
          <Box 
            component="main" 
            sx={{ 
              flexGrow: 1, 
              marginLeft: '280px',
              padding: '40px',
              minHeight: '100vh',
              maxWidth: '1600px',
              bgcolor: 'background.default'
            }}
          >
            <Routes>
              <Route path="/" element={<ManualInput />} />
              <Route path="/batch" element={<BatchUpload />} />
              <Route path="/linkedin" element={<LinkedInInput />} />
              <Route path="/candidates" element={<CandidatesPage />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App; 