import React from 'react';
import { Box, Typography, Link } from '@mui/material';

// CandidateCard-Komponente: Zeigt die wichtigsten Infos eines Kandidaten in einer Card an
const CandidateCard = ({ candidate }) => {
  // Name zusammensetzen
  const name = `${candidate.firstName || ''} ${candidate.lastName || ''}`.trim();
  
  // Konfidenz oder Wechseldatum je nach Modelltyp
  const isTimeSeriesModel = candidate.modelType === 'tft' || candidate.modelType === 'gru';
  
  // Konfidenz korrekt extrahieren
  const confidence = candidate.confidence ? candidate.confidence[0] : 0;

  // Für Zeitreihen-Modelle: Konfidenz als Tage interpretieren
  const getDaysFromConfidence = (conf) => {
    if (!conf) return 0;
    // Konfidenz in Tage umrechnen (Beispiel: 0.8 = 80 Tage)
    return Math.round(conf);
  };

  // Gibt die Farbe je nach Konfidenz/Zeit zurück
  const getConfidenceColor = (value, isTimeSeries) => {
    if (isTimeSeries) {
      // Für Zeitreihen-Modelle: 
      // - unter 6 Monaten (180 Tage) = grün
      // - 6 Monate bis 1 Jahr (180-365 Tage) = orange
      // - über 1 Jahr = rot
      const days = getDaysFromConfidence(value);
      if (days <= 180) return '#8AD265'; // grün
      if (days <= 365) return '#FFC03D'; // orange
      return '#FF2525'; // rot
    } else {
      // Für Klassifikations-Modelle: Je höher die Wahrscheinlichkeit, desto roter
      const percentage = value * 100;
      if (percentage <= 50) return '#FF2525'; // rot
      if (percentage <= 75) return '#FFC03D'; // gelb
      return '#8AD265'; // grün
    }
  };

  return (
    <Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', height: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Bild und Name/Link nebeneinander */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
        {/* Profilbild, falls vorhanden */}
        {candidate.imageUrl && candidate.imageUrl !== '' && (
          <img src={candidate.imageUrl} alt={name} style={{ width: 80, height: 80, borderRadius: '50%', objectFit: 'cover', border: '2px solid #eee' }} />
        )}
        <Box>
          {/* Name */}
          <Typography variant="h3" sx={{ fontSize: '1.2rem', fontWeight: 600, color: '#1a1a1a' }}>{name}</Typography>
          {/* LinkedIn-Profil-Link */}
          <Link href={candidate.linkedinProfile} target="_blank" rel="noopener noreferrer" sx={{ color: '#001B41', textDecoration: 'none', '&:hover': { color: '#FF8000' } }}>LinkedIn Profil</Link>
        </Box>
      </Box>

      {/* Aktuelle Position anzeigen, falls vorhanden */}
      {candidate.currentPosition && (<Typography sx={{ color: '#666', fontSize: '1rem', mt: 1 }}><b>Aktuelle Position:</b> {candidate.currentPosition}</Typography>)}
      {/* Standort anzeigen, falls vorhanden */}
      {candidate.location && (<Typography sx={{ color: '#666', fontSize: '1rem' }}><b>Standort:</b> {candidate.location}</Typography>)}
      {/* Branche anzeigen, falls vorhanden */}
      {candidate.industry && (<Typography sx={{ color: '#666', fontSize: '1rem' }}><b>Branche:</b> {candidate.industry}</Typography>)}

      {/* Wechselwahrscheinlichkeit oder Wechseldatum */}
      <Box sx={{ mt: 2 }}>
        {isTimeSeriesModel ? (
          // Zeitreihen-Modell Anzeige
          <Box sx={{ 
            display: 'flex',
            flexDirection: 'column',
            gap: 1.5,
            mb: 2 
          }}>
            <Typography sx={{ 
              color: '#666', 
              fontSize: '1rem',
              fontWeight: 600 
            }}>
              Voraussichtliches Wechseldatum
            </Typography>
            <Typography sx={{ 
              color: getConfidenceColor(confidence, true), 
              fontSize: '1.2rem',
              fontWeight: 700 
            }}>
              {new Date(Date.now() + getDaysFromConfidence(confidence) * 24 * 60 * 60 * 1000).toLocaleDateString('de-DE', {
                day: '2-digit',
                month: 'long',
                year: 'numeric'
              })}
            </Typography>
          </Box>
        ) : (
          // Klassifikations-Modell Anzeige
          <>
            <Typography sx={{ color: getConfidenceColor(confidence, false), fontWeight: 600, mb: 1 }}>
              {confidence * 100 <= 60 ? 'Geringe Wechselwahrscheinlichkeit' : confidence * 100 <= 80 ? 'Mittlere Wechselwahrscheinlichkeit' : 'Hohe Wechselwahrscheinlichkeit'}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
              <Typography sx={{ fontWeight: 600, minWidth: 50, color: getConfidenceColor(confidence, false) }}>
                {(confidence * 100).toFixed(0)}%
              </Typography>
              <Box sx={{ flexGrow: 1, height: 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
                <Box sx={{ height: '100%', width: `${confidence * 100}%`, bgcolor: getConfidenceColor(confidence, false), borderRadius: 1, transition: 'width 0.3s ease' }} />
              </Box>
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
};

export default CandidateCard; 