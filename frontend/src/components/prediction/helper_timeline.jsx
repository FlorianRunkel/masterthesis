import React, { useState } from 'react';
import { Box, Typography, Tooltip, useTheme, useMediaQuery, Collapse, IconButton, Button } from '@mui/material';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import SearchIcon from '@mui/icons-material/Search';
import EditNoteIcon from '@mui/icons-material/EditNote';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

// Color palette for SHAP bar chart (top 5 + other)
const SHAP_BAR_COLORS = [
  '#8AD265', // grün
  '#B6D94C', // gelb-grün
  '#FFD700', // gelb
  '#FFA500', // orange
  '#FF8C00', // orange
  '#FF6F00', // orange
  '#FF4500', // orange
  '#FF2525', // orange
  '#FF2525', // rot
  '#666'     // grau für "Other"
];

/**
 * Timeline component visualizes the predicted career change process for a candidate.
 * It displays the current status, expected job search phases, and the estimated job change date.
 * Additionally, it explains the prediction using a SHAP bar chart.
 *
 * @param {Object} prediction - Prediction object containing confidence and explanations.
 * @param {Object} profile - Candidate profile (currently unused, reserved for future use).
 */
const Timeline = ({ prediction, profile }) => {
  // ========== Theme & Responsiveness ==========
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // ========== User Permission Check ==========
  const user = JSON.parse(localStorage.getItem('user'));
  const canViewExplanations = user?.canViewExplanations === true;

  // ========== State für das Ausklappen der exakten Tage ==========
  const [showDays, setShowDays] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState('shap'); // 'shap' oder 'lime'

  // ========== Early Exit if No Prediction ==========
  if (!prediction) return null;
  
  // ========== Calculate Timeline Dates ==========
  // Extract confidence value (days until job change)
  const confidenceValue = Array.isArray(prediction.confidence) ? prediction.confidence[0] : prediction.confidence;
  const daysUntilChange = Math.round(confidenceValue);
  const today = new Date();
  const changeDate = new Date(today.getTime() + daysUntilChange * 24 * 60 * 60 * 1000);

  // Calculate intermediate phases (first job search, intensive search)
  const daysUntilFirstSearch = Math.round(daysUntilChange * 0.3);
  const daysUntilIntensiveSearch = Math.round(daysUntilChange * 0.7);
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

  // ===================== SHAP & LIME Explanations =====================
  /**
   * SHAP and LIME explanations show which features had the greatest impact on the prediction.
   * Zeige alle Features mit impact_percentage >= 5% einzeln, alle darunter als 'Other'.
   */
  
  // SHAP Explanations - unterstütze beide Datenstrukturen
  let shapExplanations = prediction.shap_explanations || prediction.explanations || [];
  
  // Fallback: Falls explanations ein Boolean ist, versuche andere Felder
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

  // LIME Explanations - unterstütze beide Datenstrukturen
  let limeExplanations = prediction.lime_explanations || [];
  
  // Fallback: Falls lime_explanations nicht korrekt ist
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
      color: SHAP_BAR_COLORS[9] // immer grau für Other
    });
  }

  // Wähle aktuelle Daten basierend auf Dropdown
  const currentBarData = selectedMethod === 'shap' ? shapBarData : limeBarData;
  const currentMethod = selectedMethod === 'shap' ? 'SHAP' : 'LIME';
  const hasExplanations = (selectedMethod === 'shap' && shapBarData.length > 0) || (selectedMethod === 'lime' && limeBarData.length > 0);
  
  // Verfügbare Methoden für Dropdown
  const availableMethods = [];
  if (shapBarData.length > 0) availableMethods.push('shap');
  if (limeBarData.length > 0) availableMethods.push('lime');
  
  // Setze Standard-Methode auf erste verfügbare
  if (availableMethods.length > 0 && !availableMethods.includes(selectedMethod)) {
    setSelectedMethod(availableMethods[0]);
  }
  
  // ========== Days Until Job Change (Range & Toggle) ==========
  function formatMonthsRange(start, end) {
    if (end < 12) {
      // Alles unter 12 Monaten: "X-Y Monate"
      return `${start}-${end} months`;
    } else {
      // Ab 12 Monaten: "N Jahr(e) und X-Y Monate"
      const years = Math.floor(start / 12);
      const remStart = start % 12;
      const remEnd = end % 12;
      const yearLabel = `${years} year${years === 1 ? '' : 's'}`;
      // Wenn z.B. 12-14 → 1 Jahr und 0-2 Monate
      return `${yearLabel} ${remStart}-${remEnd} months`;
    }
  }
  
  // Berechne die Range in 3-Monats-Schritten
  const daysPerMonth = 30.44;
  const months = daysUntilChange / daysPerMonth;
  const rangeStart = Math.floor((months - 1) / 3) * 3 + 1;
  const rangeEnd = rangeStart + 2;
  const rangeLabel = formatMonthsRange(rangeStart, rangeEnd);

  // ===================== Render =====================
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
      {/* ===== Timeline Explanation Header ===== */}
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

      {/* ===== Days Until Job Change (Highlight) ===== */}
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

      {/* ===== Timeline Visualization ===== */}
      <Box sx={{ width: '100%', maxWidth: 1200, mb: 2, position: 'relative', minHeight: { xs: 500, md: 180 } }}>
        {/* Horizontal line (only visible on large screens) */}
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
        {/* Timeline phases (responsive layout) */}
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
              {/* Icon and vertical line (mobile), icon on top (desktop) */}
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
                  zIndex: 2 // icon above the line
                }}
              >
                {phase.icon}
                {/* Vertical line below icon (only on mobile) */}
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
                          ? 'linear-gradient(180deg, #3B82F6 0%, #F59E42 100%)' // blue to orange
                          : idx === 1
                          ? 'linear-gradient(180deg, #F59E42 0%, #F59E42 100%)' // orange to orange
                          : 'linear-gradient(180deg, #F59E42 0%, #F87171 100%)', // orange to red
                      transform: 'translateX(-50%)',
                      zIndex: 1,
                      display: { xs: 'block', sm: 'block', md: 'block', lg: 'none' }
                    }}
                  />
                )}
              </Box>
              {/* Phase text block */}
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

      {/* ===== SHAP & LIME Explanations Bar Chart ===== */}
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
          
          {/* Buttons nur anzeigen, wenn mehr als eine Methode verfügbar ist */}
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

          {/* Erklärung IMMER anzeigen */}
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
          
          {/* Bar chart */}
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
          
          {/* Legend */}
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