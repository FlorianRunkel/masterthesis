import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

const getBarColors = [
  '#28a745', // grün
  '#ffc107', // gelb
  '#dc3545', // rot
  '#b0b0b0'  // grau für Sonstiges
];

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const recommendations = Array.isArray(prediction.recommendations) 
    ? prediction.recommendations 
    : [prediction.recommendations];

  const getProbabilityClass = (confidence) => {
    if (confidence < 60) return 'probability-low-single';
    if (confidence < 80) return 'probability-medium-single';
    return 'probability-high-single';
  };

  const confidenceValue = Array.isArray(prediction.confidence) 
    ? prediction.confidence[0] 
    : prediction.confidence;
    
  const confidence = Math.round(confidenceValue * 100);
  const probabilityClass = getProbabilityClass(confidence);

  // Feature Importances für gestapelten Balken vorbereiten
  let explanations = prediction.explanations || [];
  explanations = explanations.slice().sort((a, b) => b.impact_percentage - a.impact_percentage);
  const top3 = explanations.slice(0, 3);
  const sonstigeSumme = explanations.slice(3).reduce((sum, f) => sum + f.impact_percentage, 0);
  const barData = [
    ...top3.map((f, i) => ({
      ...f,
      color: getBarColors[i]
    })),
    ...(sonstigeSumme > 0 ? [{
      feature: 'Sonstiges',
      impact_percentage: sonstigeSumme,
      color: getBarColors[3]
    }] : [])
  ];

  return (
    <Box sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" color="primary" gutterBottom>
            Wechselwahrscheinlichkeit
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h3" sx={{ color: '#001B41', mr: 2 }}>
              {confidence}%
            </Typography>
            <Box sx={{ flex: 1, position: 'relative', height: '16px', mr: 2 }}>
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '100%',
                  width: '100%',
                  bgcolor: '#f0f0f0',
                  borderRadius: '6px',
                  overflow: 'hidden',
                }}
              />
              <Box
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '100%',
                  width: `${confidence}%`,
                  bgcolor: probabilityClass === 'probability-low-single' ? '#dc3545' :
                          probabilityClass === 'probability-medium-single' ? '#ffc107' : '#28a745',
                  borderRadius: '6px',
                  transition: 'width 0.3s ease',
                }}
              />
            </Box>
          </Box>
        </Box>

        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" color="primary" gutterBottom>
            Empfehlungen
          </Typography>
          <Box component="ul" sx={{ listStyle: 'none', p: 0, m: 0 }}>
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
                    '&:last-child': { mb: 0 }
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
      </Paper>

      {/* Vorhersage-Erklärung separat und als gestapelter Balken */}
      {barData.length > 0 && (
        <Box sx={{ mt: 4, mb: 2 }}>
          <Typography variant="h6" color="primary" gutterBottom sx={{ mb: 2, fontSize: '1.2rem' }}>
            Vorhersage-Erklärung
          </Typography>
          {/* Gestapelter Balken */}
          <Box sx={{ display: 'flex', width: '100%', height: 28, borderRadius: 2, overflow: 'hidden', boxShadow: 1, mb: 2 }}>
            {barData.map((item, idx) => (
              <Box
                key={item.feature}
                sx={{
                  width: `${item.impact_percentage}%`,
                  bgcolor: item.color,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontWeight: 600,
                  fontSize: '0.95rem',
                  borderRight: idx < barData.length - 1 ? '2px solid #fff' : 'none',
                  transition: 'width 0.3s ease'
                }}
              >
                {item.impact_percentage > 8 && `${item.impact_percentage.toFixed(1)}%`}
              </Box>
            ))}
          </Box>
          {/* Legende */}
          <Box sx={{ display: 'flex', gap: 2, mt: 1, flexWrap: 'wrap' }}>
            {barData.map(item => (
              <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: 16, height: 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                <Typography variant="body2">{item.feature}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default PredictionResult; 