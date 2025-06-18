import React from 'react';
import { Box, Typography, Tooltip, useTheme, useMediaQuery } from '@mui/material';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import SearchIcon from '@mui/icons-material/Search';
import EditNoteIcon from '@mui/icons-material/EditNote';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';

// Color palette for SHAP bar chart (top 5 + other)
const SHAP_BAR_COLORS = [
  '#8AD265', // green
  '#FFC03D', // yellow
  '#FFA500', // orange
  '#FF6F00', // dark orange
  '#FF2525', // red
  '#666'     // gray for 'Other'
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

  // ========== Timeline Phases Definition ==========
  /**
   * Each phase represents a milestone in the predicted job change process.
   * - Today: Job change detected
   * - First search: First job search activities expected
   * - Intensive search: Intensive job search phase
   * - Change: Expected job change
   */
  const phases = [
    {
      icon: <CalendarTodayIcon sx={{ fontSize: 32, color: '#3B82F6' }} />,
      label: 'Today',
      desc: 'Job change detected',
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

  // ===================== SHAP Explanations =====================
  /**
   * SHAP explanations show which features had the greatest impact on the prediction.
   * - Top 5 features are shown with distinct colors.
   * - 'Other' groups remaining features (if present).
   * - Bar chart is normalized to 100%.
   */
  let explanations = prediction.explanations || [];
  const otherFeature = explanations.find(f => f.feature === 'Other');
  const mainFeatures = explanations.filter(f => f.feature !== 'Other');
  // Sort by impact descending
  const sortedFeatures = mainFeatures.slice().sort((a, b) => b.impact_percentage - a.impact_percentage);
  // Top 5 + Other
  const barData = [
    ...sortedFeatures.slice(0, 5).map((f, i) => ({ ...f, color: SHAP_BAR_COLORS[i] || '#666' })),
    ...(otherFeature ? [{ ...otherFeature, color: SHAP_BAR_COLORS[5] || '#666' }] : [])
  ];
  // Normalize to 100%
  const totalImpact = barData.reduce((sum, item) => sum + item.impact_percentage, 0);
  const normalizedBarData = barData.map(item => ({
    ...item,
    impact_percentage: totalImpact > 0 ? (item.impact_percentage * 100) / totalImpact : 0
  }));

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
        <Typography sx={{ fontWeight: 900, fontSize: isMobile ? 40 : 54, color: '#F59E42', mb: isMobile ? 0.2 : 0.5, lineHeight: 1 }}>
          {daysUntilChange}
        </Typography>
        <Typography sx={{ fontWeight: 600, fontSize: isMobile ? '0.9rem' : '1rem', color: '#001242' }}>
          Days until job change
        </Typography>
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

      {/* ===== SHAP Explanations Bar Chart ===== */}
      {normalizedBarData.length > 0 && (
        <Box sx={{ width: '100%', mb: isMobile ? 1 : 2 }}>
          <Typography
            variant="h6"
            color="primary"
            gutterBottom
            sx={{ mb: isMobile ? 1 : 2, fontSize: isMobile ? '0.9rem' : '1.1rem', fontWeight: 900, color: '#001242' }}
          >
            Explanation of the prediction
          </Typography>
          <Typography
            sx={{ color: '#444', fontSize: isMobile ? '0.85rem' : '0.98rem', lineHeight: 1.7, textAlign: 'justify', mb: isMobile ? 1 : 2 }}
          >
            The following bar shows which features had the greatest impact on the result. The larger the colored section, the more important this feature was for the prediction. The legend below explains what the colors stand for.
          </Typography>
          {/* SHAP bar chart */}
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
            {normalizedBarData.map((item, idx) => (
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
                  borderRight: idx < normalizedBarData.length - 1 ? '2px solid #fff' : 'none',
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
          {/* SHAP legend */}
          <Box sx={{ display: 'flex', gap: isMobile ? 1 : 2, mt: isMobile ? 1 : 2, flexWrap: 'wrap' }}>
            {normalizedBarData.map(item => (
              <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Box sx={{ width: isMobile ? 12 : 16, height: isMobile ? 12 : 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                <Typography variant="body2" sx={{ fontSize: isMobile ? '0.7rem' : '0.8rem' }}>{item.feature}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default Timeline; 