import React from 'react';
import { Box, Typography, Chip } from '@mui/material';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

const zeitraumRanges = [
  { label: "0-3 Monate", start: 0, end: 90, color: "#28a745" },
  { label: "3-6 Monate", start: 91, end: 180, color: "#2ecc71" },
  { label: "6-9 Monate", start: 181, end: 270, color: "#f1c40f" },
  { label: "9-12 Monate", start: 271, end: 365, color: "#e67e22" },
  { label: "12-18 Monate", start: 366, end: 545, color: "#e74c3c" },
  { label: "über 18 Monate", start: 546, end: 730, color: "#b0b0b0" }
];

const MAE = 190.76;
const RMSE = 368.66;

function calculateAdjustedConfidence(prediction) {
  if (!prediction.predictions || !prediction.predictions[0]) return prediction.confidence && prediction.confidence[0] ? prediction.confidence[0] * MAE : 0;
  const vorhersage = prediction.predictions[0].vorhersage;
  const uncertainty = vorhersage.unsicherheit;
  const daysFromConfidence = (prediction.confidence && prediction.confidence[0] ? prediction.confidence[0] : 0) * MAE;
  const rmseFactor = 1 - (uncertainty / RMSE);
  const adjustedDays = daysFromConfidence * rmseFactor;
  return adjustedDays;
}

const PredictionResultTime = ({ prediction }) => {
  if (!prediction) return null;

  // Berechne Werte
  const adjustedDays = calculateAdjustedConfidence(prediction);
  const heute = new Date();
  const tageBisWechsel = Math.round(adjustedDays);
  const wechseldatum = new Date(heute.getTime() + tageBisWechsel * 24 * 60 * 60 * 1000);
  const wechseldatumStr = wechseldatum.toLocaleDateString('de-DE', { year: 'numeric', month: 'long', day: 'numeric' });

  // Zeitraum-Label
  let zeitraumLabel = "-";
  for (const r of zeitraumRanges) {
    if (tageBisWechsel >= r.start && tageBisWechsel <= r.end) {
      zeitraumLabel = r.label;
      break;
    }
  }
  const zeitraumColor = zeitraumRanges.find(r => r.label === zeitraumLabel)?.color || '#b0b0b0';

  // Unsicherheit
  let unsicherheit = null;
  if (prediction.predictions && prediction.predictions[0] && prediction.predictions[0].vorhersage) {
    unsicherheit = prediction.predictions[0].vorhersage.unsicherheit;
  }

  return (
    <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 6, mb: 6 }}>
      <Box sx={{
        bgcolor: '#fff',
        borderRadius: '22px',
        boxShadow: '0 4px 24px 0 rgba(0,0,0,0.10)',
        p: { xs: 3, sm: 5 },
        minWidth: { xs: '90vw', sm: 400 },
        maxWidth: 520,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 3,
        border: '1.5px solid #e0e7ef',
        minWidth: '1200px',
      }}>
        <Typography variant="h2" sx={{ fontSize: { xs: '1.3rem', sm: '1.7rem' }, fontWeight: 800, color: '#001B41', mb: 1, textAlign: 'left', letterSpacing: 0.2 }}>
          Prognose zum nächsten Jobwechsel
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
          <CalendarMonthIcon sx={{ color: '#075056', fontSize: 38 }} />
          <Typography sx={{ fontSize: { xs: '1.15rem', sm: '1.35rem' }, fontWeight: 700, color: '#075056', letterSpacing: 0.2 }}>
            {wechseldatumStr}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
          <AccessTimeIcon sx={{ color: zeitraumColor, fontSize: 32 }} />
          <Chip label={zeitraumLabel} sx={{ bgcolor: zeitraumColor, color: '#fff', fontWeight: 700, fontSize: '1.15rem', px: 2.5, py: 1, borderRadius: 2, minWidth: 120, textAlign: 'center' }} />
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
          <TrendingUpIcon sx={{ color: '#001B41', fontSize: 32 }} />
          <Typography sx={{ fontSize: '1.15rem', color: '#001B41', fontWeight: 600, letterSpacing: 0.1 }}>
            Prognose-Score: <b>{(prediction.confidence && prediction.confidence[0] ? (prediction.confidence[0]*100).toFixed(1) : '-')}%</b>
          </Typography>
        </Box>
        {unsicherheit !== null && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
            <InfoOutlinedIcon sx={{ color: '#e67e22', fontSize: 28 }} />
            <Typography sx={{ fontSize: '1.08rem', color: '#e67e22', fontWeight: 500 }}>
              Unsicherheit: {unsicherheit.toFixed(1)} Tage
            </Typography>
          </Box>
        )}
        {prediction.llm_explanation && (
          <Box sx={{ mt: 2, p: 2.5, bgcolor: '#f5f5f5', borderRadius: 2, width: '100%' }}>
            <Typography sx={{ color: '#444', fontSize: '1.13rem', lineHeight: 1.8, textAlign: 'center' }}>
              {prediction.llm_explanation}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default PredictionResultTime;
