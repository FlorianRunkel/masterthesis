import React from 'react';
import { Box, Typography } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import Timeline from './helper_timeline';

const PredictionResultTime = ({ prediction }) => {
  if (!prediction) return null;

  return (
    <Box sx={{ width: '100%', display: 'flex', mt: 4, mb: 4, justifyContent: 'center', height: '100%'}}>
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px',boxShadow: '0 2px 8px rgba(0,0,0,0.05)', p: { xs: 2, sm: 3 }, width: '100%', display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {prediction.llm_explanation && (
          <Box sx={{ 
            mt: 1, 
            mb: 3, 
            p: 2.5, 
            bgcolor: '#FFF8E1', 
            borderRadius: 3,
            border: '1px solid #FFE082',
            boxShadow: '0 2px 8px rgba(255, 193, 7, 0.1)'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1.5 }}>
              <InfoIcon sx={{ 
                color: '#444', 
                lineHeight: 1.5,
                fontSize: '1.3rem',
                display: 'flex',
                alignItems: 'center'
              }} />
              <Typography sx={{ 
                color: '#444',
                fontSize: '1.3rem',
                lineHeight: 1.5,
                textAlign: 'center',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center'
              }}>
                {prediction.llm_explanation}
              </Typography>
            </Box>
          </Box>
        )}
        <Timeline prediction={prediction} />
      </Box>
    </Box>
  );
};

export default PredictionResultTime;
