import React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import theme from './theme';
import Sidebar from './components/layout/Sidebar';
import LinkedInInput from './components/candidates/LinkedInInput';
import BatchUpload from './components/candidates/BatchUpload';
import ManualInput from './components/candidates/ManualInput';

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
              padding: '40px 0',
              minHeight: '100vh',
              maxWidth: '1600px',
              paddingLeft: '80px',
              paddingRight: '40px',
              bgcolor: 'background.default',
              position: 'relative',
              '&::before': {
                content: '""',
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                background: 'radial-gradient(circle, rgba(245, 245, 240, 0) 40%, rgba(245, 245, 240, 0.8) 70%, rgba(242, 242, 242, 1) 100%)',
                zIndex: -1,
              }
            }}
          >
            <Routes>
              <Route path="/" element={<ManualInput />} />
              <Route path="/batch" element={<BatchUpload />} />
              <Route path="/linkedin" element={<LinkedInInput />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App; 