import React from 'react';
import { Box, Typography, Link } from '@mui/material';

// CandidateCard-Komponente: Zeigt die wichtigsten Infos eines Kandidaten in einer Card an
const CandidateCard = ({ candidate }) => {
  // Name zusammensetzen
  const name = `${candidate.firstName || ''} ${candidate.lastName || ''}`.trim();
  // Konfidenz (Wechselwahrscheinlichkeit) berechnen
  const confidence = candidate.confidence ? candidate.confidence[0] * 100 : 0;

  // Gibt die Farbe je nach Konfidenz zurück (rot, gelb, grün)
  const getConfidenceColor = (confidence) => {
    if (confidence <= 50) return '#FF2525'; // rot
    if (confidence <= 75) return '#FFC03D'; // gelb
    return '#8AD265'; // grün
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

      {/* Wechselwahrscheinlichkeit als Text und Balkenanzeige */}
      <Box sx={{ mt: 2 }}>
        {/* Textliche Einschätzung */}
        <Typography sx={{ color: getConfidenceColor(confidence), fontWeight: 600, mb: 1 }}>
          {confidence <= 60 ? 'Geringe Wechselwahrscheinlichkeit' : confidence <= 80 ? 'Mittlere Wechselwahrscheinlichkeit' : 'Hohe Wechselwahrscheinlichkeit'}
        </Typography>
        {/* Prozentwert und Balkenanzeige */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
          {/* Prozentwert */}
          <Typography sx={{ fontWeight: 600, minWidth: 50, color: getConfidenceColor(confidence) }}>{confidence.toFixed(0)}%</Typography>
          {/* Balkenanzeige */}
          <Box sx={{ flexGrow: 1, height: 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
            <Box sx={{ height: '100%', width: `${confidence}%`, bgcolor: getConfidenceColor(confidence), borderRadius: 1, transition: 'width 0.3s ease' }} />
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default CandidateCard; 