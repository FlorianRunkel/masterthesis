import React, { useState, useEffect } from 'react';
import { Box, Typography, Fade, Slide } from '@mui/material';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import FlagIcon from '@mui/icons-material/Flag';
import EastIcon from '@mui/icons-material/East';

const Timeline = ({ prediction }) => {
  // Animation States (Hooks müssen immer am Anfang stehen!)
  const [showArrow, setShowArrow] = useState(false);
  const [showWechsel, setShowWechsel] = useState(false);
  const [showBadge, setShowBadge] = useState(false);

  useEffect(() => {
    setShowArrow(false); setShowWechsel(false); setShowBadge(false);
    const t1 = setTimeout(() => setShowArrow(true), 250);
    const t2 = setTimeout(() => setShowWechsel(true), 400);
    const t3 = setTimeout(() => setShowBadge(true), 700);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, [prediction]);

  if (!prediction) return null;
  const confidenceValue = Array.isArray(prediction.confidence) ? prediction.confidence[0] : prediction.confidence;
  const tageBisWechsel = Math.round(confidenceValue);
  const heute = new Date();
  const wechseldatum = new Date(heute.getTime() + tageBisWechsel * 24 * 60 * 60 * 1000);

  // Animation CSS
  const bounce = {
    animation: 'flagBounce 1.6s infinite cubic-bezier(.68,-0.55,.27,1.55)'
  };
  const arrowFly = {
    fontSize: 64,
    color: '#b0b0b0',
    background: '#fff',
    borderRadius: '50%',
    boxShadow: '0 2px 8px #001B4122',
    padding: 1,
    animation: 'arrowPulse 2s infinite ease-in-out',
    pointerEvents: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    ml: 8,
    mt: 5,
  };

  return (
    <Box sx={{ width: '100%', my: 0, p: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, position: 'relative', overflow: 'visible' }}>
      {/* Keyframes für Animationen */}
      <style>{`
        @keyframes flagBounce {
          0%, 100% { transform: translateY(0); }
          20% { transform: translateY(-8px); }
          40% { transform: translateY(0); }
          60% { transform: translateY(-6px); }
          80% { transform: translateY(0); }
        }
        @keyframes arrowPulse {
          0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.7; }
          50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
        }
      `}</style>
      <Box sx={{ position: 'relative', width: '100%', maxHeight: "420px"}}>
        <Typography variant="h6" color="primary" sx={{ fontWeight: 900, letterSpacing: 1, mb: 0.5 }}>
          Voraussichtlicher Jobwechsel
        </Typography>
        <Typography sx={{ color: '#444', fontSize: '1.1rem', lineHeight: 1.7, textAlign: 'justify', mb: 1 }}>
          Die Prognose zeigt, wann der Kandidat voraussichtlich den Job wechseln wird. Die Timeline visualisiert den Zeitraum zwischen heute und dem erwarteten Wechseldatum.
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', gap: 6, position: 'relative', zIndex: 3 }}>
          {/* Heute-Karte */}
          <Fade in={true} timeout={500}>
            <Box sx={{ bgcolor: '#001B41', color: '#fff', borderRadius: '16px', p: 4, minWidth: 140, display: 'flex', flexDirection: 'column', alignItems: 'center', boxShadow: '0 2px 8px #001B4144' }}>
              <CalendarTodayIcon sx={{ fontSize: 36, mb: 1, color: '#fff' }} />
              <Typography sx={{ fontWeight: 800, fontSize: 20, letterSpacing: 1 , color: '#fff'}}>Heute</Typography>
              <Typography sx={{ fontWeight: 500, fontSize: 18, mt: 1 , color: '#fff'}}>{heute.toLocaleDateString('de-DE')}</Typography>
            </Box>
          </Fade>
          {/* Pfeil mittig als Flex-Element */}
          <EastIcon sx={arrowFly} />
          {/* Wechsel-Karte */}
          <Slide in={showWechsel} direction="up" timeout={600}>
            <Box sx={{ bgcolor: '#FF8000', color: '#fff', borderRadius: '16px', p: 3, minWidth: 180, display: 'flex', flexDirection: 'column', alignItems: 'center', boxShadow: '0 2px 8px #FF800044' }}>
              <FlagIcon sx={{ fontSize: 36, mb: 1, color: '#fff', ...bounce }} />
              <Typography sx={{ fontWeight: 800, fontSize: 20, letterSpacing: 1, color: '#fff' }}>Wechsel</Typography>
              <Typography sx={{ fontWeight: 700, fontSize: 20, mt: 1 ,color: '#fff'}}>{wechseldatum.toLocaleDateString('de-DE')}</Typography>
              <Slide in={showBadge} direction="up" timeout={500}>
                <Typography sx={{ fontWeight: 500, fontSize: 16, mt: 1, bgcolor: '#fff', color: '#FF8000', borderRadius: 2, px: 2, py: 0.5, boxShadow: '0 1px 4px #FF800022' }}>{tageBisWechsel === 0 ? 'heute' : `in ${tageBisWechsel} Tag${tageBisWechsel === 1 ? '' : 'en'}`}</Typography>
              </Slide>
            </Box>
          </Slide>
        </Box>
      </Box>
    </Box>
  );
};

export default Timeline; 