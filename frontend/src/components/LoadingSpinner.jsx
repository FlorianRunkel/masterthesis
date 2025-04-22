import React from 'react';
import { Box, CircularProgress } from '@mui/material';

const LoadingSpinner = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '200px',
        width: '100%'
      }}
    >
      <CircularProgress 
        sx={{ 
          color: '#001B41',
          '&.MuiCircularProgress-root': {
            width: '40px !important',
            height: '40px !important'
          }
        }} 
      />
    </Box>
  );
};

export default LoadingSpinner; 