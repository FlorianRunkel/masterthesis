import React from 'react';
import { Box, Typography } from '@mui/material';
import Timeline from '../display/Timeline';

const PredictionResultTime = ({ prediction }) => {
  if (!prediction) return null;

  return (
    <Box sx={{ width: '100%', display: 'flex', mt: 4, mb: 4, justifyContent: 'center', height: '100%'}}>
      <Box sx={{ bgcolor: '#fff', borderRadius: '22px', boxShadow: '0 4px 24px 0 rgba(0,0,0,0.10)', p: { xs: 2, sm: 3 }, width: '100%', maxWidth: 1300, display: 'flex', flexDirection: 'column', gap: 1.5, border: '1.5px solid #e0e7ef'}}>
        {prediction.llm_explanation && (
          <Box sx={{ mt: 1, mb: 1, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, width: '100%' }}>
            <Typography sx={{ color: '#444', fontSize: '1.08rem', lineHeight: 1.7, textAlign: 'center' }}>
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
