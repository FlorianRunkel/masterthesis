import React from 'react';
import { Box, Typography, Paper, Tooltip, useTheme, useMediaQuery } from '@mui/material';

// --- Color palette for feature importance bars ---
const getBarColors = [
  '#8AD265', // green
  '#FFC03D', // yellow
  '#FFA500', // orange
  '#FF6F00', // dark orange
  '#FF2525', // red
  '#666'     // gray for 'Other'
];

// PredictionResult: Shows the classification prediction and explanation for a candidate
const PredictionResult = ({ prediction }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Benutzerberechtigung aus dem localStorage abrufen
  const user = JSON.parse(localStorage.getItem('user'));
  // ROBUSTE PRÜFUNG: Explizit auf den Wert `true` prüfen, um String vs. Boolean Fehler zu vermeiden.
  const canViewExplanations = user?.canViewExplanations === true;

  // --- Early return if no prediction ---
  if (!prediction) return null;

  // --- Prepare recommendations (not always used) ---
  const recommendations = Array.isArray(prediction.recommendations) ? prediction.recommendations : [prediction.recommendations];

  // --- Helper: Get probability class for styling ---
  const getProbabilityClass = (confidence) => {
    if (confidence < 40) return 'probability-low-single';  
    if (confidence < 70) return 'probability-medium-single';
    return 'probability-high-single';
  };

  // --- Calculate confidence and probability class ---
  const confidenceValue = Array.isArray(prediction.confidence) ? prediction.confidence[0] : prediction.confidence;
  const confidence = Math.round(confidenceValue * 100);
  const probabilityClass = getProbabilityClass(confidence);

  // --- Prepare feature importances for stacked bar ---
  let explanations = prediction.explanations || [];
  explanations = explanations.slice().sort((a, b) => b.impact_percentage - a.impact_percentage);
  const top5 = explanations.slice(0, 5);
  const otherSum = explanations.slice(5).reduce((sum, f) => sum + f.impact_percentage, 0);
  const barData = [
    ...top5.map((f, i) => ({ ...f, color: getBarColors[i] || '#666' })),
    ...(otherSum > 0 ? [{ feature: 'Other', impact_percentage: otherSum, color: getBarColors[5] || '#666' }] : [])
  ];
  const total = barData.reduce((sum, item) => sum + item.impact_percentage, 0);
  if (total > 0 && total !== 100) {
    barData.forEach(item => { item.impact_percentage = item.impact_percentage * 100 / total; });
  }

  // --- Main Render ---
  return (
    <Box>
      {/* --- Main Card --- */}
      <Paper elevation={3} sx={{ borderRadius: '14px', boxShadow: { xs: 4, md: 8 }, bgcolor: '#fff', p: isMobile ? 2 : 3 }}>
        {/* --- Prediction Header --- */}
        <Box sx={{ mb: isMobile ? 2 : 3 }}>
          <Typography variant="h1" color="primary" gutterBottom sx={{ fontSize: isMobile ? '1.2rem' : '1.5rem', fontWeight: 700, mb: isMobile ? 2 : 4, color: '#001B41' }}>Career Change Prediction</Typography>
          {/* --- Confidence Bar and Value --- */}
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h3" sx={{ mr: 2, fontSize: '3rem', fontWeight: 600, color: probabilityClass === 'probability-low-single' ? '#001B41' : probabilityClass === 'probability-medium-single' ? '#FFC03D' : '#8AD265' }}>{confidence}%</Typography>
            <Box sx={{ flex: 1, position: 'relative', height: '16px', mr: 2 }}>
              <Box sx={{ position: 'absolute', top: 0, left: 0, height: '100%', width: '100%', bgcolor: '#f0f0f0', borderRadius: '6px', overflow: 'hidden' }} />
              <Box sx={{ position: 'absolute', top: 0, left: 0, height: '100%', width: `${confidence}%`, bgcolor: probabilityClass === 'probability-low-single' ? '#FF2525' : probabilityClass === 'probability-medium-single' ? '#FFC03D' : '#8AD265', borderRadius: '6px', transition: 'width 0.3s ease' }} />
            </Box>
          </Box>
          {/* --- Classification Result Text --- */}
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: isMobile ? 1 : 2, mb: isMobile ? 1 : 2 }}>
            <Typography variant="h3" sx={{ fontSize: isMobile ? '1.2rem' : '1.8rem', fontWeight: 700, color: '#444' }}>
              {confidence >= 80
                ? 'Very likely to switch jobs'
                : confidence >= 60
                  ? 'Open to new opportunities'
                  : confidence >= 40
                    ? 'Not actively looking'
                    : confidence >= 20
                      ? 'Unlikely to switch jobs'
                      : 'Very unlikely to switch jobs'
              }
            </Typography>
          </Box>
        </Box>
        {/* --- Feature Importance Bar --- */}
        {canViewExplanations && barData.length > 0 && (
        <>
          <Typography variant="h6" color="primary" gutterBottom sx={{ mt: isMobile ? 1 : 2, mb: isMobile ? 0.5 : 1, fontSize: isMobile ? '0.9rem' : '1.1rem', fontWeight: 700, color: '#001B41' }}>Prediction Explanation</Typography>
          <Box sx={{ pt: 0, pb: 0 }}>
            <Typography sx={{ color: '#444', fontSize: isMobile ? '0.8rem' : '0.9rem', lineHeight: 1.9, textAlign: 'justify' }}>
              The following bar shows which features most strongly influenced the result. The larger the colored portion, the more important this feature was for the prediction. The legend below explains what the colors represent.
            </Typography>
          </Box>
          <Box sx={{ mt: isMobile ? 2 : 4, mb: isMobile ? 2 : 3 }}>
            <Box sx={{ display: 'flex', width: '100%', height: isMobile ? 24 : 32, borderRadius: 2, overflow: 'hidden', boxShadow: 1, mb: isMobile ? 1 : 2 }}>
              {barData.map((item, idx) => (
                <Box key={item.feature} sx={{ width: `${item.impact_percentage}%`, bgcolor: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 600, fontSize: isMobile ? '0.8rem' : '0.95rem', borderRight: idx < barData.length - 1 ? '2px solid #fff' : 'none', transition: 'width 0.3s ease' }}>
                  {item.impact_percentage > 8
                    ? `${item.impact_percentage.toFixed(1)}%`
                    : (<Tooltip title={`${item.impact_percentage.toFixed(1)}%`} arrow><Box sx={{ width: '100%', height: '100%' }} /></Tooltip>)}
                </Box>
              ))}
            </Box>
            <Box sx={{ display: 'flex', gap: isMobile ? 1 : 2, mt: isMobile ? 1 : 2, flexWrap: 'wrap'}}>
              {barData.map(item => (
                <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: isMobile ? 12 : 16, height: isMobile ? 12 : 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                  <Typography variant="body2" sx={{ fontSize: isMobile ? '0.7rem' : '0.8rem'}}>{item.feature}</Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </>
      )}
      {/* --- LLM Explanation (if available) --- */}
      {prediction.llm_explanation && (
        <Box sx={{ mb: isMobile ? 2 : 3, p: isMobile ? 2 : 3 }}>
          <Typography sx={{ color: '#444', fontSize: isMobile ? '0.8rem' : '0.88rem', lineHeight: 1.9, textAlign: 'justify' }}>{prediction.llm_explanation}</Typography>
        </Box>
      )}
      </Paper>
    </Box>
  );
};

export default PredictionResult; 