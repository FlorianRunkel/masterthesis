import React from 'react';
import { Box, Typography, Link, useTheme, useMediaQuery } from '@mui/material';

// CandidateCard-Komponente: Zeigt die wichtigsten Infos eines Kandidaten in einer Card an
const CandidateCard = ({ candidate }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

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
    <Box sx={{ 
      bgcolor: '#fff', 
      borderRadius: '16px', 
      p: isMobile ? '20px' : '30px', 
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)', 
      height: '95%', 
      display: 'flex', 
      flexDirection: 'column', 
      gap: isMobile ? 1 : 2, 
      overflow: 'hidden' 
    }}>
      {/* Bild und Name/Link nebeneinander */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: isMobile ? 1 : 2, mb: isMobile ? 0.5 : 1, minWidth: 0 }}>
        {/* Profilbild, falls vorhanden */}
        {candidate.imageUrl && candidate.imageUrl !== '' && (
          <img 
            src={candidate.imageUrl} 
            alt={name} 
            style={{ 
              width: isMobile ? 60 : 80, 
              height: isMobile ? 60 : 80, 
              borderRadius: '50%', 
              objectFit: 'cover', 
              border: '2px solid #eee', 
              flexShrink: 0 
            }} 
          />
        )}
        <Box sx={{ minWidth: 0 }}>
          {/* Name */}
          <Typography variant="h3" sx={{ 
            fontSize: isMobile ? '1rem' : '1.2rem', 
            fontWeight: 600, 
            color: '#1a1a1a', 
            wordBreak: 'break-word', 
            maxWidth: '100%', 
            whiteSpace: 'normal' 
          }}>
            {name}
          </Typography>
          {/* LinkedIn-Profil-Link */}
          <Link 
            href={candidate.linkedinProfile} 
            target="_blank" 
            rel="noopener noreferrer" 
            sx={{
              fontSize: isMobile ? '0.7rem' : '0.8rem', 
              color: '#001B41', 
              textDecoration: 'none', 
              '&:hover': { color: '#FF8000' }, 
              wordBreak: 'break-all', 
              maxWidth: '100%', 
              display: 'block' 
            }}
          >
            LinkedIn Profil
          </Link>
        </Box>
      </Box>

      {/* Aktuelle Position anzeigen, falls vorhanden */}
      {candidate.currentPosition && (
        <Typography sx={{ 
          color: '#666', 
          fontSize: isMobile ? '0.85rem' : '1rem', 
          mt: isMobile ? 0.5 : 1, 
          wordBreak: 'break-word', 
          maxWidth: '100%', 
          whiteSpace: 'normal' 
        }}>
          <b>Current Position:</b> {candidate.currentPosition}
        </Typography>
      )}
      {/* Standort anzeigen, falls vorhanden */}
      {candidate.location && (
        <Typography sx={{ 
          color: '#666', 
          fontSize: isMobile ? '0.85rem' : '1rem', 
          wordBreak: 'break-word', 
          maxWidth: '100%', 
          whiteSpace: 'normal' 
        }}>
          <b>Location:</b> {candidate.location}
        </Typography>
      )}
      {/* Branche anzeigen, falls vorhanden */}
      {candidate.industry && (
        <Typography sx={{ 
          color: '#666', 
          fontSize: isMobile ? '0.85rem' : '1rem', 
          wordBreak: 'break-word', 
          maxWidth: '100%', 
          whiteSpace: 'normal' 
        }}>
          <b>Industry:</b> {candidate.industry}
        </Typography>
      )}

      {/* Wechselwahrscheinlichkeit oder Wechseldatum */}
      <Box sx={{ mt: isMobile ? 1 : 2 }}>
        {isTimeSeriesModel ? (
          // Zeitreihen-Modell Anzeige
          <Box sx={{ 
            display: 'flex',
            flexDirection: 'column',
            gap: isMobile ? 1 : 1.5,
            mb: isMobile ? 1 : 2 
          }}>
            <Typography sx={{ 
              color: '#666', 
              fontSize: isMobile ? '0.85rem' : '1rem',
              fontWeight: 600 
            }}>
              Anticipated Change Date
            </Typography>
            <Typography sx={{ 
              color: getConfidenceColor(confidence, true), 
              fontSize: isMobile ? '0.95rem' : '1.1rem',
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
            <Typography sx={{ 
              color: getConfidenceColor(confidence, false), 
              fontWeight: 600, 
              mb: isMobile ? 0.5 : 1, 
              fontSize: isMobile ? '0.85rem' : '1rem'
            }}>
              {confidence * 100 <= 60 ? 'Unlikely to consider a change' : confidence * 100 <= 80 ? 'Might consider a change' : 'Likely to consider a change'}
            </Typography>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: isMobile ? 1 : 1.5, 
              mb: isMobile ? 1 : 2 
            }}>
              <Typography sx={{ 
                fontWeight: 600, 
                minWidth: isMobile ? 40 : 50, 
                color: getConfidenceColor(confidence, false), 
                fontSize: isMobile ? '0.8rem' : '0.95rem' 
              }}>
                {(confidence * 100).toFixed(0)}%
              </Typography>
              <Box sx={{ 
                flexGrow: 1, 
                height: isMobile ? 6 : 8, 
                bgcolor: '#eee', 
                borderRadius: 1, 
                overflow: 'hidden' 
              }}>
                <Box sx={{ 
                  height: '100%', 
                  width: `${confidence * 100}%`, 
                  bgcolor: getConfidenceColor(confidence, false), 
                  borderRadius: 1, 
                  transition: 'width 0.3s ease' 
                }} />
              </Box>
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
};

export default CandidateCard; 