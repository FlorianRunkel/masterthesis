import React from 'react';
import { Box, Typography } from '@mui/material';

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const recommendations = Array.isArray(prediction.recommendations) 
    ? prediction.recommendations 
    : [prediction.recommendations];

  const getProbabilityClass = (confidence) => {
    if (confidence < 50) return 'probability-low-single';
    if (confidence < 75) return 'probability-medium-single';
    return 'probability-high-single';
  };

  const confidenceValue = Array.isArray(prediction.confidence) 
    ? prediction.confidence[0] 
    : prediction.confidence;
    
  const confidence = Math.round(confidenceValue * 100);
  const probabilityClass = getProbabilityClass(confidence);

  return (
    <Box sx={{
      bgcolor: '#fff',
      borderRadius: '16px',
      p: '30px',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      mb: 4
    }}>
      <Typography variant="h2" sx={{
        fontSize: '1.5rem',
        fontWeight: 600,
        color: '#1a1a1a',
        mb: 3
      }}>
        Analyse-Ergebnisse
      </Typography>

      <Box sx={{ mb: 4 }}>
        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2
        }}>
          <Typography sx={{
            fontSize: '1.1rem',
            fontWeight: 500,
            color: '#1a1a1a'
          }}>
            Wahrscheinlichkeit der Vorhersage
          </Typography>
          <Typography sx={{
            fontWeight: 600,
            fontSize: '1.1rem',
            color: '#1a1a1a'
          }}>
            {Math.round(confidence)}%
          </Typography>
        </Box>

        <Box sx={{
          width: '100%',
          height: '8px',
          bgcolor: '#f0f0f0',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <Box
            className={probabilityClass}
            sx={{
              height: '100%',
              width: `${confidence}%`,
              bgcolor: probabilityClass === 'probability-low-single' ? '#dc3545' :
                      probabilityClass === 'probability-medium-single' ? '#ffc107' : '#28a745',
              transition: 'width 0.3s ease'
            }}
          />
        </Box>
      </Box>

      <Box>

        <Box component="ul" sx={{
          listStyle: 'none',
          p: 0,
          m: 0
        }}>
          {recommendations && recommendations.length > 0 ? (
            recommendations.map((rec, index) => (
              <Box
                component="li"
                key={index}
                sx={{
                  p: '12px 15px',
                  bgcolor: '#f5f5f5',
                  borderRadius: '8px',
                  mb: 1,
                  fontSize: '0.95rem',
                  color: '#666',
                  '&:last-child': {
                    mb: 0
                  }
                }}
              >
                {rec}
              </Box>
            ))
          ) : (
            <Box
              component="li"
              sx={{
                p: '12px 15px',
                bgcolor: '#f5f5f5',
                borderRadius: '8px',
                fontSize: '0.95rem',
                color: '#666'
              }}
            >
              Keine Empfehlungen verfügbar
            </Box>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default PredictionResult; 