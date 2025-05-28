import React from 'react';
import { Container, Typography, Box, Paper } from '@mui/material';
import ManualInput from '../components/candidates/ManualInput';

const Index = () => {
  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%)'
    }}>
      {/* Hero Section */}
      <Box 
        sx={{
          pt: { xs: 4, sm: 6, md: 8 },
          pb: { xs: 4, sm: 5, md: 6 },
          textAlign: 'center',
          background: 'linear-gradient(180deg, #0a1929 0%, #1a365d 100%)',
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%)',
            pointerEvents: 'none'
          }
        }}
      >
        <Container maxWidth="md">
          <Typography
            component="h1"
            variant="h2"
            sx={{
              fontWeight: 700,
              mb: { xs: 1, sm: 2 },
              fontSize: { xs: '2rem', sm: '2.5rem', md: '3rem' },
              background: 'linear-gradient(45deg, #fff 30%, #e0e7ff 90%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}
          >
            Career Prediction
          </Typography>
          <Typography
            variant="h5"
            color="grey.300"
            sx={{
              mb: { xs: 2, sm: 3, md: 4 },
              maxWidth: '800px',
              mx: 'auto',
              lineHeight: 1.6,
              fontSize: { xs: '1rem', sm: '1.1rem', md: '1.25rem' },
              px: { xs: 2, sm: 0 }
            }}
          >
            Analyze the probability of a single candidate's career change based on their work experience.
          </Typography>
        </Container>
      </Box>

      {/* Content Section */}
      <Container 
        maxWidth="lg" 
        sx={{ 
          py: { xs: 3, sm: 4, md: 6 },
          px: { xs: 2, sm: 3, md: 4 },
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '50%',
            transform: 'translateX(-50%)',
            width: '100vw',
            height: '100%',
            backgroundColor: 'rgba(255, 255, 255, 0.7)',
            backdropFilter: 'blur(10px)',
            zIndex: -1
          }
        }}
      >
        <Paper 
          elevation={0}
          sx={{
            p: { xs: 2, sm: 3, md: 4 },
            borderRadius: { xs: 2, sm: 3, md: 4 },
            background: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(20px)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.3)'
          }}
        >
          <ManualInput />
        </Paper>
      </Container>
    </Box>
  );
};

export default Index; 