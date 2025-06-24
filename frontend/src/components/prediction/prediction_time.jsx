import React from 'react';
import { Box, Typography } from '@mui/material';
import Timeline from './helper_timeline';  

const PredictionResultTime = ({ prediction }) => {
  if (!prediction) return null;

  return (
    <Box sx={{ width: '100%', display: 'flex', mt: 4, mb: 4, justifyContent: 'center', height: '100%'}}>
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px',boxShadow: { xs: 4, md: 8 }, p: { xs: 2, sm: 3 }, width: '100%', display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        {prediction.llm_explanation && (
          <Box sx={{ mt: 1, mb: 1, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, width: '100%' }}>
            <Typography sx={{ color: '#444', fontSize: '1rem', lineHeight: 1.7, textAlign: 'center' }}>
              {prediction.llm_explanation}
            </Typography>
          </Box>
        )}
        <Timeline prediction={prediction} />
      </Box>
    </Box>
  );
};

export default PredictionResultTime;
