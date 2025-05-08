import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link, Paper, Chip, Tooltip, IconButton } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const zeitraumRanges = [
  { label: "0-6 Monate", start: 0, end: 6, color: "#28a745" },
  { label: "7-12 Monate", start: 7, end: 12, color: "#ffc107" },
  { label: "13-24 Monate", start: 13, end: 24, color: "#dc3545" },
  { label: "über 24 Monate", start: 25, end: 36, color: "#b0b0b0" }
];

const ResultsTableTime = ({ results, onSave, isSaving }) => {
  const [selectedCandidates, setSelectedCandidates] = useState(new Set());
  const [expandedRows, setExpandedRows] = useState(new Set());

  if (!results) return null;

  const handleSelectCandidate = (index) => {
    const newSelected = new Set(selectedCandidates);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedCandidates(newSelected);
  };

  const handleSaveSelected = () => {
    const candidatesToSave = Array.from(selectedCandidates).map(index => ({
      ...results[index],
      savedAt: new Date().toISOString()
    }));
    onSave(candidatesToSave);
  };

  if (results.error) {
    let errorMessage = results.error;
    if (typeof results.error === 'string' && results.error.includes('recommendations')) {
      errorMessage = 'Fehler bei der Verarbeitung: Empfehlungen konnten nicht generiert werden.';
    }
    return (
      <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
        <Box sx={{ p: '30px', width: '100%' }}>
          <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#1a1a1a', mb: 2 }}>
            Fehler bei der Verarbeitung
          </Typography>
          <Typography sx={{ color: '#666', mb: 1 }}>{errorMessage}</Typography>
          {results.message && (
            <Typography sx={{ color: '#666' }}>{results.message}</Typography>
          )}
          {results.requirements && (
            <>
              <Typography sx={{ mt: 2, mb: 1, color: '#666' }}>Erforderliche Spalten:</Typography>
              <ul style={{ color: '#666', margin: 0, paddingLeft: 20 }}>
                {results.requirements.map((req, index) => (
                  <li key={index}>{req}</li>
                ))}
              </ul>
            </>
          )}
        </Box>
      </Box>
    );
  }

  const successCount = results.filter(r => !r.error && r.status !== 'error').length;
  const errorCount = results.filter(r => r.error || r.status === 'error').length;

  const toggleDetails = (index) => {
    const newExpandedRows = new Set(expandedRows);
    if (newExpandedRows.has(index)) {
      newExpandedRows.delete(index);
    } else {
      newExpandedRows.add(index);
    }
    setExpandedRows(newExpandedRows);
  };

  return (
    <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
      <Box sx={{
        bgcolor: '#fff',
        borderRadius: '16px',
        overflow: 'hidden',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        mb: 4,
        width: '100%'
      }}>
        <Box sx={{ p: '30px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#1a1a1a', mb: 2 }}>
              Zusammenfassung der Batch-Verarbeitung
            </Typography>
            <Typography sx={{ mb: 1, color: '#666' }}>
              <strong>Erfolgreich verarbeitet:</strong> {successCount} Kandidaten
            </Typography>
            <Typography sx={{ color: '#666' }}>
              <strong>Fehler:</strong> {errorCount} Kandidaten
            </Typography>
          </div>
          {selectedCandidates.size > 0 && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleSaveSelected}
              disabled={isSaving}
              startIcon={isSaving ? <CircularProgress size={24} sx={{ color: 'white' }} /> : <SaveIcon />}
              sx={{
                bgcolor: '#001B41',
                color: 'white',
                p: '10px 20px',
                borderRadius: '8px',
                textTransform: 'none',
                fontWeight: 600,
                '&:hover': {
                  bgcolor: '#FF5F00'
                }
              }}
            >
              {isSaving ? 'Speichern...' : selectedCandidates.size + ' Kandidaten speichern'}
            </Button>
          )}
        </Box>

        <Box sx={{ overflowX: 'auto', width: '100%' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 900,
                  fontSize: '1.1rem',
                  width: '40px'
                }}></th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 900,
                  fontSize: '1.1rem'
                }}>Wechselzeitraum</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 900,
                  fontSize: '1.1rem'
                }}>Name</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 600,
                  fontSize: '1.1rem'
                }}>LinkedIn</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight:900,
                  fontSize: '1.1rem'
                }}></th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, index) => {
                const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Nicht angegeben';
                const linkedin = result.linkedinProfile || 'Nicht angegeben';

                if (result.error || result.status === 'error') {
                  return (
                    <tr key={index} style={{ background: 'rgba(220, 53, 69, 0.05)' }}>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}></td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Chip
                          label={result.recommendations}
                          sx={{
                            bgcolor: result.recommendations ? zeitraumRanges.find(r => r.label === result.recommendations).color : 'grey',
                            color: '#fff',
                            fontWeight: 700,
                            fontSize: '1.05rem',
                            px: 2,
                            py: 0.5,
                            borderRadius: 2
                          }}
                        />
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Link 
                          href={linkedin} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          sx={{
                            color: '#666',
                            textDecoration: 'none',
                            fontSize: '0.85rem',
                            opacity: 0.8,
                            transition: 'opacity 0.2s ease',
                            '&:hover': {
                              opacity: 1
                            }
                          }}
                        >
                          {linkedin}
                        </Link>
                      </td>
                      <td colSpan="2" style={{ padding: '15px 30px', borderBottom: '1px solid #eee', color: '#dc3545' }}>
                        {result.error || 'Fehler bei der Verarbeitung'}
                      </td>
                    </tr>
                  );
                }

                // Zeitraum-Objekt finden (jetzt nach recommendations)
                const range = zeitraumRanges.find(r => r.label === result.recommendations) || zeitraumRanges[3];

                // Labels und Farben
                const labels = ["0-6 Monate", "7-12 Monate", "13-24 Monate", "über 24 Monate"];
                const confidences = result.confidence || [];
                const topIdx = confidences.indexOf(Math.max(...confidences));
                const confidencesCopy = [...confidences];
                confidencesCopy[topIdx] = -1;
                const secondIdx = confidencesCopy.indexOf(Math.max(...confidencesCopy));
                const secondLabel = labels[secondIdx];
                const secondProb = confidences[secondIdx];
                const secondRange = zeitraumRanges[secondIdx];

                return (
                  <React.Fragment key={index}>
                    <tr
                      style={{
                        transition: 'background 0.2s',
                        cursor: 'pointer',
                        ':hover': { background: '#f5f8ff' }
                      }}
                    >
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee', textAlign: 'center' }}>
                        <Checkbox
                          checked={selectedCandidates.has(index)}
                          onChange={() => handleSelectCandidate(index)}
                          sx={{
                            color: '#666',
                            '&.Mui-checked': {
                              color: '#666',
                            },
                          }}
                        />
                      </td>

                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee', fontWeight: 500, fontSize: '1.08rem' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Link
                          href={linkedin}
                          target="_blank"
                          rel="noopener noreferrer"
                          sx={{
                            color: '#001B41',
                            fontWeight: 500,
                            fontSize: '1rem',
                            textDecoration: 'none',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            '&:hover': { color: '#FF5F00', textDecoration: 'underline' }
                          }}
                        >
                          {linkedin.replace(/^https?:\/\/|^www\./, '')}
                        </Link>
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Chip
                          label={result.recommendations}
                          sx={{
                            bgcolor: range.color,
                            color: '#fff',
                            fontWeight: 700,
                            fontSize: '1.05rem',
                            px: 2,
                            py: 0.5,
                            borderRadius: 2
                          }}
                        />
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee', textAlign: 'center' }}>
                        <Button
                          onClick={() => toggleDetails(index)}
                          sx={{
                            bgcolor: '#001B41',
                            color: 'white',
                            textTransform: 'none',
                            px: 2.5,
                            py: 1.2,
                            borderRadius: '8px',
                            fontSize: '1rem',
                            fontWeight: 600,
                            minWidth: 0,
                            '&:hover': {
                              bgcolor: '#FF5F00'
                            },
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1
                          }}
                          endIcon={expandedRows.has(index) ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        >
                          {expandedRows.has(index) ? 'Details ausblenden' : 'Details anzeigen'}
                        </Button>
                      </td>
                    </tr>
                    {expandedRows.has(index) && (
                      <tr>
                        <td colSpan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}>
                          <Box sx={{ borderRadius: '16px', p: '30px', margin: '20px auto', bgcolor: '#fff', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)', maxWidth: '95%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
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
                                Zweitwahrscheinlichkeit: {secondLabel} ({(secondProb * 100).toFixed(0)}%)
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
                            {result.llm_explanation && (
                              <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
                                <Typography sx={{ color: '#444', fontSize: '1.1rem', lineHeight: 1.9 }}>
                                  {result.llm_explanation}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </Box>
      </Box>
    </Box>
  );
};

export default ResultsTableTime;
