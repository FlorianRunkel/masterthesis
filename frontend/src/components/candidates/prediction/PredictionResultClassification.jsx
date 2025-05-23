import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

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
      <Paper elevation={3} sx={{ p: 3, borderRadius: '18px' , boxShadow: '0 2px 8px rgba(0,0,0,0.04)'}}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h1" color="primary" gutterBottom sx={{fontSize: '1.8rem', fontWeight: 700, mb: 4}}>
            Wechselwahrscheinlichkeit
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h3" sx={{ color: '#001B41', mr: 2 ,fontSize: '4rem', fontWeight: 600}}>
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
          <Typography variant="h6" color="primary" gutterBottom sx={{ mt: 8, mb: 4, fontSize: '1.rem', fontWeight: 700}}>
            Vorhersage-Erklärung
          </Typography>
          <Box sx={{ p: 3, pt: 0, pb: 0 }}>
            <Typography sx={{ color: '#444', fontSize: '1.1rem', lineHeight: 1.9, textAlign: 'justify', mb: 2 }}>
              Der folgende Balken zeigt, welche Merkmale das Ergebnis am stärksten beeinflusst haben. Je größer der farbige Anteil, desto wichtiger war dieses Merkmal für die Prognose. Die Legende darunter erklärt, wofür die Farben stehen.
            </Typography>
          </Box>
          <Box sx={{ mt: 6, mb: 3, px: { xs:1, sm: 2, md: 4 } }}>
            {/* Gestapelter Balken */}
            <Box sx={{ display: 'flex', width: '100%', height: 32, borderRadius: 2, overflow: 'hidden', boxShadow: 1, mb: 2 }}>
              {barData.map((item, idx) => (
                <Box
                  key={item.feature}
                  sx={{ width: `${item.impact_percentage}%`, bgcolor: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 600, fontSize: '0.95rem', borderRight: idx < barData.length - 1 ? '2px solid #fff' : 'none', transition: 'width 0.3s ease' }}
                >
                  {item.impact_percentage > 8 && `${item.impact_percentage.toFixed(1)}%`}
                </Box>
              ))}
            </Box>
            {/* Legende */}
            <Box sx={{ display: 'flex', gap: 2, mt: 2, flexWrap: 'wrap' }}>
              {barData.map(item => (
                <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 16, height: 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                  <Typography variant="body2">{item.feature}</Typography>
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
          <Typography sx={{ color: '#444', fontSize: '1.1rem', lineHeight: 1.9, textAlign: 'justify' }}>
            {prediction.llm_explanation}
          </Typography>
        </Box>
      )}
      </Paper>


    </Box>
  );
};

export default PredictionResult; 