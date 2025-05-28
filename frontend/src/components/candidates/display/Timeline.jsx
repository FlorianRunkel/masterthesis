import React from 'react';
import { Box, Typography } from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import WorkOutlineIcon from '@mui/icons-material/WorkOutline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import FlagIcon from '@mui/icons-material/Flag';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';

const getBarColors = [
  '#8AD265', // grün
  '#FFC03D', // gelb
  '#FF2525', // rot
  '#666'  // grau für Sonstiges
];

const Timeline = ({ prediction, profile }) => {
  if (!prediction) return null;

  // Werte berechnen
  const confidenceValue = Array.isArray(prediction.confidence) ? prediction.confidence[0] : prediction.confidence;
  const tageBisWechsel = Math.round(confidenceValue);
  const heute = new Date();
  const wechselDatum = new Date(heute.getTime() + tageBisWechsel * 24 * 60 * 60 * 1000);

  // Phasen berechnen
  const tageBisBewerbung = Math.round(tageBisWechsel * 0.3);
  const tageBisIntensiveSuche = Math.round(tageBisWechsel * 0.7);
  const bewerbungsDatum = new Date(heute.getTime() + tageBisBewerbung * 24 * 60 * 60 * 1000);
  const intensiveSucheDatum = new Date(heute.getTime() + tageBisIntensiveSuche * 24 * 60 * 60 * 1000);

  // Timeline-Phasen
  const phases = [
    {
      icon: <PersonIcon sx={{ fontSize: 32, color: '#3B82F6' }} />,
      label: 'Today', 
      desc: 'Job change detected',  
      date: heute.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#3B82F6'
    },
    {
      icon: <WorkOutlineIcon sx={{ fontSize: 32, color: '#F59E42' }} />,
      label: `In about ${tageBisBewerbung} days`,
      desc: 'First job search activities expected',
      date: bewerbungsDatum.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#F59E42'
    },
    {
      icon: <TrendingUpIcon sx={{ fontSize: 32, color: '#F59E42' }} />,
      label: `In about ${tageBisIntensiveSuche} days`,
      desc: 'Intensive job search phase',
      date: intensiveSucheDatum.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#F59E42'
    },
    {
      icon: <FlagIcon sx={{ fontSize: 32, color: '#F87171' }} />,
      label: wechselDatum.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      desc: 'Expected job change',
      date: wechselDatum.toLocaleDateString('de-DE', { day: '2-digit', month: 'short', year: 'numeric' }),
      color: '#F87171'
    }
  ];

  // SHAP-Explanations
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

  return (
    <Box sx={{ width: '100%', my: 0, p: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, position: 'relative', overflow: 'visible' }}>

      {/* Erklärung über der Timeline */}
      <Box sx={{ width: '100%', mb: 0
       }}>
        <Typography variant="h6" sx={{ fontWeight: 800, color: '#13213C', mb: 1, fontSize: '1.5rem' }}>
          Career Change Prediction Timeline
        </Typography>
        <Typography sx={{ color: '#444', fontSize: '1rem', lineHeight: 1.7 }}>
          This timeline visualizes the predicted career change process for the candidate. It shows the current status, the expected start of job search activities, the phase of intensive job seeking, and the estimated date of the next job change.
        </Typography>
      </Box>
      {/* Tage bis Wechsel ganz oben */}
      <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', mb: 1, justifyContent: 'left', alignItems: 'center', gap: 1 }}>

        <Typography sx={{ fontWeight: 900, fontSize: 54, color: '#F59E42', mb: 0.5, lineHeight: 1 }}>
          {tageBisWechsel}
        </Typography>
        <Typography sx={{ fontWeight: 600, fontSize: '1rem', color: '#13213C' }}>
          Days until job change
        </Typography>
      </Box>

      {/* Timeline */}
      <Box sx={{ width: '100%', maxWidth: 1200, mb: 2, position: 'relative' }}>
        {/* Linie */}
        <Box sx={{
          position: 'absolute',
          top: 36,
          left: 0,
          right: 0,
          height: 6,
          background: 'linear-gradient(90deg, #3B82F6 0%, #F59E42 50%, #F87171 100%)',
          borderRadius: 3,
          zIndex: 1
        }} />
        {/* Phasen */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', position: 'relative', zIndex: 2 }}>
          {phases.map((phase, idx) => (
            <Box key={idx} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 180 }}>
              <Box sx={{
                bgcolor: '#fff',
                borderRadius: 3,
                boxShadow: '0 4px 16px #0001',
                p: 2,
                mb: 1,
                mt: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                minWidth: 70,
                minHeight: 70
              }}>
                {phase.icon}
              </Box>
              <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', color: phase.color, mb: 0.2 }}>{phase.label}</Typography>
              <Typography sx={{ fontWeight: 400, fontSize: '0.95rem', color: '#222', mb: 0.2 }}>{phase.desc}</Typography>
              <Typography sx={{ fontWeight: 400, fontSize: '0.9rem', color: '#888' }}>{phase.date}</Typography>
            </Box>
          ))}
        </Box>
      </Box>
      {/* SHAP-Explanations */}
      {barData.length > 0 && (
        <Box sx={{ width: '100%', mb: 2 }}>
          <Typography variant="h6" color="primary" gutterBottom sx={{ mb: 2, fontSize: '1.1rem', fontWeight: 700 }}>
            Explanation of the prediction
          </Typography>
          <Typography sx={{ color: '#444', fontSize: '0.98rem', lineHeight: 1.7, textAlign: 'justify', mb: 2 }}>
            The following bar shows which features had the greatest impact on the result. The larger the colored section, the more important this feature was for the prediction. The legend below explains what the colors stand for.
          </Typography>
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
                <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>{item.feature}</Typography>
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default Timeline; 