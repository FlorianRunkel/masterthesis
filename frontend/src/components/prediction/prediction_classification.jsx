import React, { useState } from 'react';
import { Box, Typography, Paper, Tooltip, useTheme, useMediaQuery, Button } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const SHAP_BAR_COLORS = [
  '#8AD265',
  '#B6D94C',
  '#FFD700',
  '#FFA500',
  '#FF8C00',
  '#FF6F00',
  '#FF4500',
  '#FF2525',
  '#FF2525',
  '#666'     // gray for "Other"
];

const PredictionResult = ({ prediction }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const user = JSON.parse(localStorage.getItem('user'));
  const canViewExplanations = user?.canViewExplanations === true;
  const [selectedMethod, setSelectedMethod] = useState('shap');

  if (!prediction) return null;

  const getProbabilityClass = (confidence) => {
    if (confidence < 40) return 'probability-low-single';  
    if (confidence < 70) return 'probability-medium-single';
    return 'probability-high-single';
  };

  const confidenceValue = Array.isArray(prediction.confidence) ? prediction.confidence[0] : prediction.confidence;
  const confidence = Math.round(confidenceValue * 1000) / 10;
  const probabilityClass = getProbabilityClass(confidence);
  let shapExplanations = prediction.shap_explanations || prediction.explanations || [];

  if (typeof shapExplanations === 'boolean' || !Array.isArray(shapExplanations)) {
    console.log('SHAP explanations is not an array, trying fallback...');
    shapExplanations = [];
  }

  const shapMainFeatures = shapExplanations
    .filter(f => f.impact_percentage >= 5)
    .sort((a, b) => b.impact_percentage - a.impact_percentage);
  const shapOtherFeatures = shapExplanations.filter(f => f.impact_percentage > 0 && f.impact_percentage < 10);
  const shapOtherImpact = shapOtherFeatures.reduce((sum, f) => sum + f.impact_percentage, 0);

  const shapBarData = shapMainFeatures.map((f, i) => ({
    ...f,
    color: SHAP_BAR_COLORS[i % SHAP_BAR_COLORS.length]
  }));
  if (shapOtherImpact > 0) {
    shapBarData.push({
      feature: 'Other',
      impact_percentage: shapOtherImpact,
      description: 'All features with < 10% impact',
      color: SHAP_BAR_COLORS[9]
    });
  }

  let limeExplanations = prediction.lime_explanations || [];
  if (!Array.isArray(limeExplanations)) {
    console.log('LIME explanations is not an array, using empty array...');
    limeExplanations = [];
  }

  const limeMainFeatures = limeExplanations
    .filter(f => f.impact_percentage >= 5)
    .sort((a, b) => b.impact_percentage - a.impact_percentage);
  const limeOtherFeatures = limeExplanations.filter(f => f.impact_percentage > 0 && f.impact_percentage < 10);
  const limeOtherImpact = limeOtherFeatures.reduce((sum, f) => sum + f.impact_percentage, 0);

  const limeBarData = limeMainFeatures.map((f, i) => ({
    ...f,
    color: SHAP_BAR_COLORS[i % SHAP_BAR_COLORS.length]
  }));
  if (limeOtherImpact > 0) {
    limeBarData.push({
      feature: 'Other',
      impact_percentage: limeOtherImpact,
      description: 'All features with < 10% impact',
      color: SHAP_BAR_COLORS[9]
    });
  }

  const currentBarData = selectedMethod === 'shap' ? shapBarData : limeBarData;
  const currentMethod = selectedMethod === 'shap' ? 'SHAP' : 'LIME';
  const hasExplanations = (selectedMethod === 'shap' && shapBarData.length > 0) || (selectedMethod === 'lime' && limeBarData.length > 0);
  const availableMethods = [];
  if (shapBarData.length > 0) availableMethods.push('shap');
  if (limeBarData.length > 0) availableMethods.push('lime');
  if (availableMethods.length > 0 && !availableMethods.includes(selectedMethod)) {
    setSelectedMethod(availableMethods[0]);
  }
  return (
    <Box>
      <Paper elevation={3} sx={{ borderRadius: '14px', boxShadow: '0 2px 8px rgba(0,0,0,0.05)', bgcolor: '#fff', p: isMobile ? 2 : 3 }}>

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
              <InfoIcon sx={{ color: '#444', fontSize: '1.3rem' }} />
              <Typography sx={{ 
                color: '#444', 
                fontSize: '1rem', 
                lineHeight: 1.7, 
                textAlign: 'center',
                fontWeight: 500
              }}>
                {prediction.llm_explanation}
              </Typography>
            </Box>
          </Box>
        )}

        <Box sx={{ mb: isMobile ? 2 : 3 }}>
          <Typography variant="h1" color="primary" gutterBottom sx={{ fontSize: isMobile ? '1.2rem' : '1.5rem', fontWeight: 700, color: '#001B41' }}>Career Change Prediction</Typography>
          <Typography sx={{ color: '#444', fontSize: '0.95rem', mb: isMobile ? 2 : 4 }}>
            The candidate has been classified by the XGBoost model with a predicted probability of job change.
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h3" sx={{ mr: 2, fontSize: '3rem', fontWeight: 600, color: probabilityClass === 'probability-low-single' ? '#001B41' : probabilityClass === 'probability-medium-single' ? '#FFC03D' : '#8AD265' }}>{confidence}%</Typography>
            <Box sx={{ flex: 1, position: 'relative', height: '16px', mr: 2 }}>
              <Box sx={{ position: 'absolute', top: 0, left: 0, height: '100%', width: '100%', bgcolor: '#f0f0f0', borderRadius: '6px', overflow: 'hidden' }} />
              <Box sx={{ position: 'absolute', top: 0, left: 0, height: '100%', width: `${confidence}%`, bgcolor: probabilityClass === 'probability-low-single' ? '#FF2525' : probabilityClass === 'probability-medium-single' ? '#FFC03D' : '#8AD265', borderRadius: '6px', transition: 'width 0.3s ease' }} />
            </Box>
          </Box>
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
        {canViewExplanations && hasExplanations && (
        <>
          <Typography variant="h6" color="primary" gutterBottom sx={{ mt: isMobile ? 1 : 2, mb: isMobile ? 0.5 : 1, fontSize: isMobile ? '0.9rem' : '1.1rem', fontWeight: 700, color: '#001B41' }}>Prediction Explanation</Typography>
          {availableMethods.length > 1 && (
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-start', gap: isMobile ? 1 : 2 }}>
              <Button
                onClick={() => setSelectedMethod('shap')}
                variant={selectedMethod === 'shap' ? 'contained' : 'outlined'}
                sx={{
                  borderRadius: '12px',
                  fontWeight: 700,
                  fontSize: isMobile ? '0.7rem' : '0.9rem',
                  minWidth: isMobile ? 60 : 80,
                  px: isMobile ? 0.5 : 1,
                  py: isMobile ? 0.5 : 0.8,
                  bgcolor: selectedMethod === 'shap' ? '#EB7836' : '#fff',
                  color: selectedMethod === 'shap' ? '#fff' : '#EB7836',
                  borderColor: '#EB7836',
                  boxShadow: selectedMethod === 'shap' ? '0 2px 8px #EB783633' : 'none',
                  transition: 'all 0.2s',
                  '&:hover': {
                    bgcolor: '#001242',
                    color: '#fff',
                    borderColor: '#fff',
                  }
                }}
              >
                SHAP
              </Button>
              <Button
                onClick={() => setSelectedMethod('lime')}
                variant={selectedMethod === 'lime' ? 'contained' : 'outlined'}
                sx={{
                  borderRadius: '12px',
                  fontWeight: 700,
                  fontSize: isMobile ? '0.7rem' : '0.9rem',
                  minWidth: isMobile ? 60 : 80,
                  px: isMobile ? 0.5 : 1,
                  py: isMobile ? 0.5 : 0.8,
                  bgcolor: selectedMethod === 'lime' ? '#EB7836' : '#fff',
                  color: selectedMethod === 'lime' ? '#fff' : '#EB7836',
                  borderColor: '#EB7836',
                  boxShadow: selectedMethod === 'lime' ? '0 2px 8px #fff' : 'none',
                  transition: 'all 0.2s',
                  '&:hover': {
                    bgcolor: '#001242',
                    color: '#fff',
                    borderColor: '#fff',
                  }
                }}
              >
                LIME
              </Button>
            </Box>
          )}
          <Typography
            sx={{ color: '#666', fontSize: isMobile ? '0.8rem' : '1rem', lineHeight: 1.7, mb: isMobile ? 1.2 : 2, textAlign: 'justify'}}
          >
            {currentMethod === 'SHAP' 
              ? 'SHAP (SHapley Additive exPlanations) explains model predictions by fairly distributing the impact of each input feature, based on game theory. It considers all possible combinations of features to estimate how much each one contributes to the final prediction. This provides a consistent and theoretically grounded way to understand the influence of each feature.'
              : 'LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the complex model locally with a simpler, interpretable one. It generates small variations of the input and observes how the prediction changes, helping to understand which features were most important for that specific prediction.'
            }
          </Typography>

          <Box sx={{ pt: 0, pb: 0 }}>
            <Typography sx={{ color: '#444', fontSize: isMobile ? '0.8rem' : '0.9rem', lineHeight: 1.9, textAlign: 'justify' }}>
              The following bar shows which features most strongly influenced the result. The larger the colored portion, the more important this feature was for the prediction. The legend below explains what the colors represent.
            </Typography>
          </Box>
          <Box sx={{ mt: isMobile ? 2 : 4, mb: isMobile ? 2 : 3 }}>
            <Box sx={{ display: 'flex', width: '100%', height: isMobile ? 24 : 32, borderRadius: 2, overflow: 'hidden', boxShadow: 1, mb: isMobile ? 1 : 2 }}>
              {currentBarData.map((item, idx) => (
                <Box key={item.feature} sx={{ width: `${item.impact_percentage}%`, bgcolor: item.color, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontWeight: 600, fontSize: isMobile ? '0.8rem' : '0.95rem', borderRight: idx < currentBarData.length - 1 ? '2px solid #fff' : 'none', transition: 'width 0.3s ease' }}>
                  {item.impact_percentage > 8
                    ? `${item.impact_percentage.toFixed(1)}%`
                    : (<Tooltip title={`${item.impact_percentage.toFixed(1)}%`} arrow><Box sx={{ width: '100%', height: '100%' }} /></Tooltip>)}
                </Box>
              ))}
            </Box>
            <Box sx={{ display: 'flex', gap: isMobile ? 1 : 2, mt: isMobile ? 1 : 2, flexWrap: 'wrap'}}>
              {currentBarData.map(item => (
                <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: isMobile ? 12 : 16, height: isMobile ? 12 : 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                  <Typography variant="body2" sx={{ fontSize: isMobile ? '0.7rem' : '0.8rem'}}>{item.feature}</Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </>
      )}
      </Paper>
    </Box>
  );
};

export default PredictionResult; 