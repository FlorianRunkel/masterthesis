import React from 'react';
import { Box, Typography } from '@mui/material';

const zeitraumRanges = [
    { label: "0-6 Monate", start: 0, end: 6, color: "#28a745" },
    { label: "7-12 Monate", start: 7, end: 12, color: "#ffc107" },
    { label: "13-24 Monate", start: 13, end: 24, color: "#dc3545" },
    { label: "über 24 Monate", start: 25, end: 36, color: "#b0b0b0" }
  ];

const PredictionResultTime = ({ prediction }) => {
  if (!prediction) return null;

  // Labels und Farben
  const labels = ["0-6 Monate", "7-12 Monate", "13-24 Monate", "über 24 Monate"];
  const confidences = Array.isArray(prediction.confidence) ? prediction.confidence : [0,0,0,0];
  const topIdx = confidences.indexOf(Math.max(...confidences));
  const confidencesCopy = [...confidences];
  confidencesCopy[topIdx] = -1;
  const secondIdx = confidencesCopy.indexOf(Math.max(...confidencesCopy));
  const secondLabel = labels[secondIdx];
  const secondProb = confidences[secondIdx];
  const range = zeitraumRanges[topIdx] || zeitraumRanges[3];
  const secondRange = zeitraumRanges[secondIdx];

  return (
    <Box sx={{ borderRadius: '16px', p: '30px', margin: '20px auto', bgcolor: '#fff', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)', maxWidth: '100%', display: 'flex', flexDirection: 'column'}}>
      
        <Typography 
          variant="h4" 
          sx={{ 
            fontWeight: 800, 
            color: '#001B41', 
            mb: 0.5, 
            textAlign: 'left'
          }}
        >
          Wechselwahrscheinlichkeit
        </Typography>
        <Typography 
          variant="subtitle1" 
          sx={{ 
            color: '#888', 
            mb: 5, 
            textAlign: 'left',
            fontSize: '1.05rem',
            fontWeight: 400
          }}
        >
          Prognose für die Wechselbereitschaft des Kandidaten
        </Typography>
      {/* Zeitstrahl */}
      <Box sx={{ width: '100%', mx: 'auto', mt: 1, mb: 1, position: 'relative', height: 28 }}>
        {/* Zeitstrahl */}
        <Box sx={{
          height: 24,
          borderRadius: 12,
          background: '#fff',
          border: '2px solid #e0e0e0',
          width: '100%',
          position: 'absolute',
          top: 2,
          left: 0,
          zIndex: 1
        }} />
        {/* Zweitwahrscheinlichkeit Marker */}
        {secondIdx !== topIdx && (
          <Box
            sx={{
              position: 'absolute',
              left: `${(secondIdx * 25)}%`,
              width: '25%',
              height: 24,
              top: 2,
              bgcolor: secondRange.color,
              borderRadius: 12,
              opacity: 0.25,
              zIndex: 2,
              pointerEvents: 'none'
            }}
          />
        )}
        {/* Hauptwahrscheinlichkeit Marker */}
        <Box
          sx={{
            position: 'absolute',
            left: `calc(${topIdx * 25}% - 1px)`,
            width: 'calc(25% + 2px)',
            height: 24,
            top: 2,
            bgcolor: range.color,
            borderRadius: 12,
            opacity: 0.85,
            zIndex: 3
          }}
        />
        {/* Achsenbeschriftung */}
        <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-between', fontSize: 15, color: '#888', fontWeight: 500, position: 'absolute', top: 28 }}>
          <span style={{ width: '25%', textAlign: 'center' }}>6</span>
          <span style={{ width: '25%', textAlign: 'center' }}>12</span>
          <span style={{ width: '25%', textAlign: 'center' }}>24</span>
          <span style={{ width: '25%', textAlign: 'center' }}>36+</span>
        </Box>
      </Box>
      {/* Zweitwahrscheinlichkeit */}
      {secondIdx !== topIdx && (
        <Typography sx={{ color: secondRange.color, textAlign: 'center', fontSize: '1rem', mt: 5, fontWeight: 500 }}>
          Zweitwahrscheinlichkeit: {secondLabel}
        </Typography>
      )}
      {/* Empfehlungstext */}
      <Typography sx={{ mt: 2, color: '#444', textAlign: 'center', fontSize: '1.08rem' }}>
        {range.label === "0-6 Monate" && "Jetzt ist der ideale Zeitpunkt für eine Ansprache!"}
        {range.label === "7-12 Monate" && "In den nächsten Monaten könnte ein Wechsel interessant werden."}
        {range.label === "13-24 Monate" && "Mittelfristig beobachten, noch kein akuter Wechselbedarf."}
        {range.label === "über 24 Monate" && "Aktuell wenig Wechselbereitschaft, langfristig in Kontakt bleiben."}
      </Typography>
      {/* LLM-Erklärung */}
      {prediction.llm_explanation && (
        <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
          <Typography sx={{ color: '#444', fontSize: '1.1rem', lineHeight: 1.9 }}>
            {prediction.llm_explanation}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default PredictionResultTime;
