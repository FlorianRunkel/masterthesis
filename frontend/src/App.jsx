import React from 'react';
import { ThemeProvider, useTheme } from '@mui/material/styles';
import { CssBaseline, Box, useMediaQuery } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import theme from './theme';
import Sidebar from './components/layout/Sidebar';
import LinkedInInput from './components/candidates/forms/LinkedInInput';
import BatchUpload from './components/candidates/forms/BatchUpload';
import ManualInput from './components/candidates/forms/ManualInput';
import CandidatesPage from './pages/CandidatesPage';

const App = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Sidebar />
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
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
};

export default App; 