import React from 'react';
import { Box, Typography, Paper, Tooltip } from '@mui/material';

const getBarColors = [
  '#8AD265', // grün
  '#FFC03D', // gelb
  '#FF2525', // rot
  '#666'  // grau für Sonstiges
];

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const recommendations = Array.isArray(prediction.recommendations) 
    ? prediction.recommendations 
    : [prediction.recommendations];

  const getProbabilityClass = (confidence) => {
    if (confidence < 60) return 'probability-low-single';
    if (confidence < 85) return 'probability-medium-single';
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
      feature: 'Other',
      impact_percentage: sonstigeSumme,
      color: getBarColors[3]
    }] : [])
  ];

  const total = barData.reduce((sum, item) => sum + item.impact_percentage, 0);
  if (total > 0 && total !== 100) {
    barData.forEach(item => {
      item.impact_percentage = item.impact_percentage * 100 / total;
    });
  }

  return (
    <Box>
      <Paper elevation={3} sx={{borderRadius: '12px' , boxShadow: '0 4px 18px 0 rgba(0,0,0,0.04)', bgcolor: '#fff', p: 3}}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h1" color="primary" gutterBottom sx={{fontSize: '1.5rem', fontWeight: 700, mb: 4, color: '#001B41'}}>
          Career Change Prediction
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h3" sx={{ mr: 2 ,fontSize: '3rem', fontWeight: 600, color: probabilityClass === 'probability-low-single' ? '#001B41' : probabilityClass === 'probability-medium-single' ? '#FFC03D' : '#8AD265'}}>
              {confidence}%
            </Typography>
            <Box sx={{ flex: 1, position: 'relative', height: '16px', mr: 2 }}>
              <Box
                sx={{position: 'absolute',  top: 0,  left: 0,  height: '100%', width: '100%',  bgcolor: '#f0f0f0',  borderRadius: '6px', overflow: 'hidden', }} />
              <Box
                sx={{position: 'absolute', top: 0, left: 0,  height: '100%',  width: `${confidence}%`, bgcolor: probabilityClass === 'probability-low-single' ? '#FF2525' : probabilityClass === 'probability-medium-single' ? '#FFC03D' : '#8AD265', borderRadius: '6px', transition: 'width 0.3s ease',}} />
            </Box>
          </Box>
        </Box>
        {barData.length > 0 && (
        <>
          <Typography variant="h6" color="primary" gutterBottom sx={{ mt: 2, mb: 1, fontSize: '1.1rem', fontWeight: 700, color: '#001B41'}}>
            Prediction Explanation
          </Typography>
          <Box sx={{ pt: 0, pb: 0 }}>
            <Typography sx={{ color: '#444', fontSize: '0.9rem', lineHeight: 1.9, textAlign: 'justify'}}>
              The following bar shows which features most strongly influenced the result. The larger the colored portion, the more important this feature was for the prediction. The legend below explains what the colors represent.
            </Typography>
          </Box>
          <Box sx={{ mt: 4, mb: 3 }}>
            {/* Gestapelter Balken */}
            <Box sx={{ display: 'flex', width: '100%', height: 32, borderRadius: 2, overflow: 'hidden', boxShadow: 1, mb: 2 }}>
              {barData.map((item, idx) => (
                <Box
                  key={item.feature}
                  sx={{ width: `${item.impact_percentage}%`, bgcolor: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 600, fontSize: '0.95rem', borderRight: idx < barData.length - 1 ? '2px solid #fff' : 'none', transition: 'width 0.3s ease' }}
                >
                  {item.impact_percentage > 8
                    ? `${item.impact_percentage.toFixed(1)}%`
                    : (
                        <Tooltip title={`${item.impact_percentage.toFixed(1)}%`} arrow>
                          <Box sx={{ width: '100%', height: '100%' }} />
                        </Tooltip>
                      )
                  }
                </Box>
              ))}
            </Box>
            {/* Legende */}
            <Box sx={{ display: 'flex', gap: 2, mt: 2, flexWrap: 'wrap'}}>
              {barData.map(item => (
                <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 16, height: 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                  <Typography variant="body2" sx={{ fontSize: '0.8rem'}}>{item.feature}</Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </>
      )}

      {/* KI-Erklärung anzeigen, falls vorhanden */}
      {prediction.llm_explanation && (
        <Box sx={{
          mb: 3,
          p: 3,
        }}>
          <Typography sx={{ color: '#444', fontSize: '0.88rem', lineHeight: 1.9, textAlign: 'justify' }}>
            {prediction.llm_explanation}
          </Typography>
        </Box>
      )}
      </Paper>


    </Box>
  );
};

export default PredictionResult; 