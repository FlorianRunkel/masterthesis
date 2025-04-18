import React from 'react';
import { FormControl, InputLabel, Select, MenuItem, Paper, Typography, Box, FormHelperText } from '@mui/material';

const ModelSelection = ({ selectedModel, onModelChange, error }) => {
  return (
    <Paper 
      elevation={0} 
      sx={{ 
        width: '100%',
        bgcolor: 'white',
        borderRadius: 2,
        p: 3
      }}
    >
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Modellauswahl
        </Typography>
      </Box>
      
      <FormControl fullWidth error={!!error}>
        <InputLabel>Vorhersagemodell</InputLabel>
        <Select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          label="Vorhersagemodell"
          sx={{
            bgcolor: 'white',
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: error ? 'error.main' : 'rgba(0, 0, 0, 0.23)'
            }
          }}
        >
          <MenuItem value="GRU">GRU (Gated Recurrent Unit)</MenuItem>
          <MenuItem value="TFT">TFT (Temporal Fusion Transformer)</MenuItem>
          <MenuItem value="XGBoost">XGBoost</MenuItem>
        </Select>
        {error && (
          <FormHelperText error>
            {error}
          </FormHelperText>
        )}
      </FormControl>
    </Paper>
  );
};

export default ModelSelection; 