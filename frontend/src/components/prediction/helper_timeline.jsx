import React, { useState } from 'react';
import { Box, Typography, Tooltip, useTheme, useMediaQuery, Collapse, IconButton, Button } from '@mui/material';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import SearchIcon from '@mui/icons-material/Search';
import EditNoteIcon from '@mui/icons-material/EditNote';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

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
  '#666'
];

const Timeline = ({ prediction, profile }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const user = JSON.parse(localStorage.getItem('user'));
  const canViewExplanations = user?.canViewExplanations === true;
  const [showDays, setShowDays] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState('shap');

  if (!prediction) return null;

  const confidenceValue = Array.isArray(prediction.confidence) ? prediction.confidence[0] : prediction.confidence;
  const daysUntilChange = prediction.predicted_time_to_change || Math.round(confidenceValue);
  const today = new Date();
  const changeDate = new Date(today.getTime() + daysUntilChange * 24 * 60 * 60 * 1000);

  const maxFirstSearchLead = 180; // 6 Months
  const maxIntensiveLead = 90;    // 3 Months

  const daysUntilFirstSearch = Math.max(daysUntilChange - maxFirstSearchLead, 0);
  const daysUntilIntensiveSearch = Math.max(daysUntilChange - maxIntensiveLead, 0);

  const firstSearchDate = new Date(today.getTime() + daysUntilFirstSearch * 24 * 60 * 60 * 1000);
  const intensiveSearchDate = new Date(today.getTime() + daysUntilIntensiveSearch * 24 * 60 * 60 * 1000);

  const phases = [
    {
      icon: <CalendarTodayIcon sx={{ fontSize: 32, color: '#3B82F6' }} />,
      label: 'Today',
      desc: '',
      date: today.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#3B82F6'
    },
    {
      icon: <SearchIcon sx={{ fontSize: 32, color: '#F59E42' }} />,
      label: `In about ${daysUntilFirstSearch} days`,
      desc: 'First job search activities expected',
      date: firstSearchDate.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#F59E42'
    },
    {
      icon: <EditNoteIcon sx={{ fontSize: 32, color: '#F59E42' }} />,
      label: `In about ${daysUntilIntensiveSearch} days`,
      desc: 'Intensive job search phase',
      date: intensiveSearchDate.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#F59E42'
    },
    {
      icon: <SwapHorizIcon sx={{ fontSize: 32, color: '#F87171' }} />,
      label: changeDate.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      desc: 'Expected job change',
      date: changeDate.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#F87171'
    }
  ];

  let shapExplanations = prediction.shap_explanations || prediction.explanations || [];
  if (typeof shapExplanations === 'boolean' || !Array.isArray(shapExplanations)) {
    console.log('SHAP explanations is not an array, trying fallback...');
    shapExplanations = [];
  }

  const shapMainFeatures = shapExplanations
    .filter(f => f.impact_percentage >= 5)
    .sort((a, b) => b.impact_percentage - a.impact_percentage);
  const shapOtherFeatures = shapExplanations.filter(f => f.impact_percentage > 0 && f.impact_percentage < 5);
  const shapOtherImpact = shapOtherFeatures.reduce((sum, f) => sum + f.impact_percentage, 0);

  const shapBarData = shapMainFeatures.map((f, i) => ({
    ...f,
    color: SHAP_BAR_COLORS[i % SHAP_BAR_COLORS.length]
  }));
  if (shapOtherImpact > 0) {
    shapBarData.push({
      feature: 'Other',
      impact_percentage: shapOtherImpact,
      description: 'All features with < 5% impact',
      color: SHAP_BAR_COLORS[9] // immer grau für Other
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
  const limeOtherFeatures = limeExplanations.filter(f => f.impact_percentage > 0 && f.impact_percentage < 5);
  const limeOtherImpact = limeOtherFeatures.reduce((sum, f) => sum + f.impact_percentage, 0);

  const limeBarData = limeMainFeatures.map((f, i) => ({
    ...f,
    color: SHAP_BAR_COLORS[i % SHAP_BAR_COLORS.length]
  }));
  if (limeOtherImpact > 0) {
    limeBarData.push({
      feature: 'Other',
      impact_percentage: limeOtherImpact,
      description: 'All features with < 5% impact',
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

  function formatTimeRange(days) {
    const months = days / 30.44;
    const years = Math.floor(months / 12);
    const remainingMonths = Math.floor(months % 12);
    
    if (years === 0) {
      return `${remainingMonths} months`;
    } else if (remainingMonths === 0) {
      return `${years} year${years === 1 ? '' : 's'}`;
    } else {
      return `${years} year${years === 1 ? '' : 's'} ${remainingMonths} months`;
    }
  }

  function formatRangeLabel(days) {
    const months = days / 30.44;
    const years = Math.floor(months / 12);
    const remainingMonths = Math.floor(months % 12);
    
    // Flexible Range: 3-4 Monate je nach Position
    let startMonth, endMonth;
    
    if (remainingMonths <= 1) {
      // Am Anfang: 0-3 Monate
      startMonth = 0;
      endMonth = 3;
    } else if (remainingMonths >= 10) {
      // Am Ende: 9-11 Monate
      startMonth = 9;
      endMonth = 11;
    } else {
      // In der Mitte: ±1.5 Monate (3-4 Monate Range)
      startMonth = Math.max(0, Math.floor(remainingMonths - 1.5));
      endMonth = Math.min(11, Math.ceil(remainingMonths + 1.5));
    }
    
    if (years === 0) {
      // Wenn startMonth 0 ist, zeige nur endMonth
      if (startMonth === 0) {
        return `${endMonth} months`;
      } else {
        return `${startMonth}-${endMonth} months`;
      }
    } else {
      const yearLabel = `${years} year${years === 1 ? '' : 's'}`;
      // Wenn startMonth 0 ist, zeige nur endMonth
      if (startMonth === 0) {
        return `${yearLabel} ${endMonth} months`;
      } else {
        return `${yearLabel} ${startMonth}-${endMonth} months`;
      }
    }
  }

  const rangeLabel = formatRangeLabel(daysUntilChange);

  return (
    <Box
      sx={{
        width: '100%',
        my: 0,
        p: isMobile ? 0.5 : 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: isMobile ? 2 : 4,
        position: 'relative',
        overflow: 'visible'
      }}
    >
      <Box sx={{ width: '100%', mb: 0 }}>
        <Typography
          variant="h6"
          sx={{ fontWeight: 800, color: '#001242', mb: isMobile ? 0.5 : 1, fontSize: isMobile ? '1.2rem' : '1.5rem' }}
        >
          Career Change Prediction Timeline
        </Typography>
        <Typography
          sx={{ color: '#444', fontSize: isMobile ? '0.9rem' : '1rem', lineHeight: 1.7 }}
        >
          This timeline visualizes the predicted career change process for the candidate. It shows the current status, the expected start of job search activities, the phase of intensive job seeking, and the estimated date of the next job change.
        </Typography>
      </Box>
      <Box
        sx={{
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          mb: isMobile ? 0.5 : 1,
          justifyContent: 'left',
          alignItems: 'center',
          gap: isMobile ? 0.5 : 1
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography sx={{ fontWeight: 900, fontSize: isMobile ? 40 : 54, color: '#F59E42', mb: isMobile ? 0.2 : 0.5, lineHeight: 1 }}>
            {rangeLabel}
          </Typography>
          <IconButton size="small" onClick={() => setShowDays(v => !v)} aria-label="Show exact days">
            <ExpandMoreIcon sx={{ transform: showDays ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
          </IconButton>
        </Box>
        <Collapse in={showDays}>
          <Typography sx={{ fontWeight: 600, fontSize: isMobile ? '0.9rem' : '1rem', color: '#001242', mt: 0.5 }}>
            {daysUntilChange} days until job change
          </Typography>
        </Collapse>
      </Box>
      <Box sx={{ width: '100%', maxWidth: 1200, mb: 2, position: 'relative', minHeight: { xs: 500, md: 180 } }}>
        <Box
          sx={{
            position: 'absolute',
            top: 36,
            left: 0,
            right: 0,
            width: '100%',
            height: 6,
            background: 'linear-gradient(90deg, #3B82F6 0%, #F59E42 50%, #F87171 100%)',
            borderRadius: 3,
            zIndex: 1,
            display: { xs: 'none', sm: 'none', md: 'none', lg: 'block' }
          }}
        />
        <Box
          sx={{
            display: { xs: 'flex', sm: 'flex', md: 'flex', lg: 'grid' },
            flexDirection: { xs: 'column', sm: 'column', md: 'column', lg: 'unset' },
            gridTemplateColumns: { lg: 'repeat(auto-fit, minmax(0px, 1fr))' },
            position: 'relative',
            zIndex: 2,
            gap: { xs: 4, sm: 4, md: 4, lg: 2 },
            width: '100%'
          }}
        >
          {phases.map((phase, idx) => (
            <Box
              key={idx}
              sx={{
                display: 'flex',
                flexDirection: { xs: 'row', sm: 'row', md: 'row', lg: 'column' },
                alignItems: { xs: 'flex-start', sm: 'flex-start', md: 'flex-start', lg: 'center' },
                justifyContent: 'center',
                minWidth: { xs: 0, lg: 180 },
                width: { xs: '100%', lg: '100%' },
                mb: { xs: 2, sm: 2, md: 2, lg: 0 },
                gap: { xs: 2, sm: 2, md: 2, lg: 0 },
                position: 'relative',
                flexWrap: { xs: 'nowrap', sm: 'nowrap', md: 'nowrap', lg: 'unset' }
              }}
            >
              <Box
                sx={{
                  bgcolor: '#fff',
                  borderRadius: 3,
                  boxShadow: '0 4px 16px #0001',
                  transform: { xs: 'translateX(100%)', sm: 'translateX(100%)', md: 'translateX(140%)', lg: 'none' },
                  p: 2,
                  mb: { xs: 0, sm: 0, md: 0, lg: 1 },
                  mt: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  minWidth: 70,
                  minHeight: 70,
                  position: 'relative',
                  zIndex: 2
                }}
              >
                {phase.icon}
                {idx < phases.length - 1 && (
                  <Box
                    sx={{
                      position: 'absolute',
                      left: '50%',
                      top: '100%',
                      width: 4,
                      height: 40,
                      background:
                        idx === 0
                          ? 'linear-gradient(180deg, #3B82F6 0%, #F59E42 100%)'
                          : idx === 1
                          ? 'linear-gradient(180deg, #F59E42 0%, #F59E42 100%)'
                          : 'linear-gradient(180deg, #F59E42 0%, #F87171 100%)',
                      transform: 'translateX(-50%)',
                      zIndex: 1,
                      display: { xs: 'block', sm: 'block', md: 'block', lg: 'none' }
                    }}
                  />
                )}
              </Box>
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: { xs: 'flex-start', sm: 'flex-start', md: 'flex-start', lg: 'center' },
                  justifyContent: 'center',
                  transform: { xs: 'translateX(20%)', lg: 'none' },
                  minWidth: 0,
                  flex: 1,
                  textAlign: { xs: 'left', sm: 'left', md: 'left', lg: 'center' },
                  ml: { xs: 2, sm: 2, md: 2, lg: 0 }
                }}
              >
                <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', color: phase.color, mb: 0.2 }}>{phase.label}</Typography>
                <Typography sx={{ fontWeight: 400, fontSize: '0.95rem', color: '#222', mb: 0.2 }}>{phase.desc}</Typography>
                <Typography sx={{ fontWeight: 400, fontSize: '0.9rem', color: '#888' }}>{phase.date}</Typography>
              </Box>
            </Box>
          ))}
        </Box>
      </Box>
      {canViewExplanations && (
        <Box sx={{ width: '100%', mb: isMobile ? 1 : 2 }}>
          <Typography
            variant="h6"
            color="primary"
            gutterBottom
            sx={{ mb: isMobile ? 1 : 2, fontSize: isMobile ? '0.9rem' : '1.1rem', fontWeight: 900, color: '#001242' }}
          >
            Explanation of the prediction
          </Typography>
          {hasExplanations && availableMethods.length > 1 && (
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
          {hasExplanations && (
            <Typography
              sx={{ color: '#666', fontSize: isMobile ? '0.8rem' : '1rem', lineHeight: 1.7, mb: isMobile ? 1.2 : 2, textAlign: 'justify'}}
            >
              {currentMethod === 'SHAP' 
                ? 'SHAP (SHapley Additive exPlanations) explains model predictions by fairly distributing the impact of each input feature, based on game theory. It considers all possible combinations of features to estimate how much each one contributes to the final prediction. This provides a consistent and theoretically grounded way to understand the influence of each feature.'
                : 'LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by approximating the complex model locally with a simpler, interpretable one. It generates small variations of the input and observes how the prediction changes, helping to understand which features were most important for that specific prediction.'
              }
            </Typography>
          )}
          <Box
            sx={{
              display: 'flex',
              width: '100%',
              height: isMobile ? 24 : 32,
              borderRadius: 2,
              overflow: 'hidden',
              boxShadow: 1,
              mb: isMobile ? 1 : 2
            }}
          >
            {currentBarData.map((item, idx) => (
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
                  fontSize: isMobile ? '0.8rem' : '0.95rem',
                  borderRight: idx < currentBarData.length - 1 ? '2px solid #fff' : 'none',
                  transition: 'width 0.3s ease'
                }}
              >
                {item.impact_percentage > 8 ? (
                  `${item.impact_percentage.toFixed(1)}%`
                ) : (
                  <Tooltip title={`${item.impact_percentage.toFixed(1)}%`} arrow>
                    <Box sx={{ width: '100%', height: '100%' }} />
                  </Tooltip>
                )}
              </Box>
            ))}
          </Box>
          <Box sx={{ display: 'flex', gap: isMobile ? 1 : 2, mt: isMobile ? 1 : 2, flexWrap: 'wrap' }}>
            {currentBarData.map(item => (
              <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: isMobile ? 12 : 16, height: isMobile ? 12 : 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                <Tooltip title={item.feature} arrow>
                  <Typography variant="body2" sx={{ fontSize: isMobile ? '0.7rem' : '0.8rem' }}>{item.feature}</Typography>
                </Tooltip>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default Timeline; 