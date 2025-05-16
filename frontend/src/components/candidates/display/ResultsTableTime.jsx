import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link, Paper, Chip, Tooltip, IconButton } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const zeitraumRanges = [
  { label: "0-3 Monate", start: 0, end: 90, color: "#28a745" },
  { label: "3-6 Monate", start: 91, end: 180, color: "#2ecc71" },
  { label: "6-9 Monate", start: 181, end: 270, color: "#f1c40f" },
  { label: "9-12 Monate", start: 271, end: 365, color: "#e67e22" },
  { label: "12-18 Monate", start: 366, end: 545, color: "#e74c3c" },
  { label: "über 18 Monate", start: 546, end: 730, color: "#b0b0b0" }
];

const ResultsTableTime = ({ results, onSave, isSaving }) => {
  const [selectedCandidates, setSelectedCandidates] = useState(new Set());
  const [expandedRows, setExpandedRows] = useState(new Set());

  // Konstanten für die Metriken
  const MAE = 190.76; // Tage
  const RMSE = 368.66; // Tage

  // Funktion zur Bestimmung des Zeitraums basierend auf Tagen
  const getTimeRangeFromDays = (days) => {
    if (days <= 90) return "0-3 Monate";
    if (days <= 180) return "3-6 Monate";
    if (days <= 270) return "6-9 Monate";
    if (days <= 365) return "9-12 Monate";
    if (days <= 545) return "12-18 Monate";
    return "über 18 Monate";
  };

  // Funktion zur Berechnung des angepassten Confidence-Scores
  const calculateAdjustedConfidence = (result) => {
    if (!result.predictions || !result.predictions[0]) return result.confidence;

    const prediction = result.predictions[0].vorhersage;
    const median = prediction.median;
    const uncertainty = prediction.unsicherheit;
    
    // Berechne die tatsächlichen Tage basierend auf der Confidence und MAE
    const daysFromConfidence = result.confidence * MAE;
    
    // Berechne den Unsicherheitsfaktor basierend auf RMSE
    const rmseFactor = 1 - (uncertainty / RMSE);
    
    // Kombiniere die Faktoren
    const adjustedDays = daysFromConfidence * rmseFactor;
    
    return adjustedDays;
  };

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
    if (typeof results.error === 'string' && results.error.includes('confidence')) {
      errorMessage = 'Processing error: Recommendations could not be generated.';
    }
    return (
      <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
        <Box sx={{ p: '30px', width: '100%' }}>
          <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#1a1a1a', mb: 2 }}>
            Processing error
          </Typography>
          <Typography sx={{ color: '#666', mb: 1 }}>{errorMessage}</Typography>
          {results.message && (
            <Typography sx={{ color: '#666' }}>{results.message}</Typography>
          )}
          {results.requirements && (
            <>
              <Typography sx={{ mt: 2, mb: 1, color: '#666' }}>Required columns:</Typography>
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
              Batch Processing Summary
            </Typography>
            <Typography sx={{ mb: 1, color: '#666' }}>
              <strong>Successfully processed:</strong> {successCount} candidates
            </Typography>
            <Typography sx={{ color: '#666' }}>
              <strong>Errors:</strong> {errorCount} candidates
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
              {isSaving ? 'Saving...' : selectedCandidates.size + ' Save candidates'}
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
                }}>Name</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 900,
                  fontSize: '1.1rem'
                }}>LinkedIn</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 900,  
                  fontSize: '1.1rem'
                }}>Job change period</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 900,
                  fontSize: '1.1rem'
                }}></th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, index) => {
                const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Not specified';
                const linkedin = result.linkedinProfile || 'Not specified';

                if (result.error || result.status === 'error') {
                  return (
                    <tr key={index} style={{ background: 'rgba(220, 53, 69, 0.05)' }}>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}></td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Chip
                          label={result.confidence}
                          sx={{
                            bgcolor: result.confidence ? zeitraumRanges.find(r => r.label === result.confidence).color : 'grey',
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
                        {result.error || 'Processing error'}
                      </td>
                    </tr>
                  );
                }

                const adjustedDays = calculateAdjustedConfidence(result);
                const heute = new Date();
                const tageBisWechsel = Math.round(adjustedDays * MAE);
                const wechseldatum = new Date(heute.getTime() + tageBisWechsel * 24 * 60 * 60 * 1000);
                
                // Dynamische Anpassung: Startdatum flexibel, aber nie weniger als 4% vom linken Rand
                const maxShortRangeDays = 180; // 6 Monate
                let startDate, endDate;
                if (tageBisWechsel < maxShortRangeDays) {
                  // Bereich mittig: 1 Monat vor heute, 1 Monat nach Wechsel
                  startDate = new Date(heute.getFullYear(), heute.getMonth() - 1, heute.getDate());
                  endDate = new Date(wechseldatum.getFullYear(), wechseldatum.getMonth() + 1, wechseldatum.getDate());
                } else {
                  // Standardbereich: heute bis X Monate nach heute
                  startDate = new Date();
                  endDate = new Date(startDate.getFullYear(), startDate.getMonth() + Math.max(6, Math.ceil(tageBisWechsel / 30) + 1), 1);
                }
                const totalDays = (endDate - startDate) / (1000 * 60 * 60 * 24);
                // Offset für linken Rand, damit heute nie ganz links klebt
                const leftOffset = 4;
                const rightOffset = 100;
                // todayPos relativ zu startDate, aber nie weniger als leftOffset
                let todayPos = ((heute - startDate) / (endDate - startDate)) * (rightOffset - leftOffset) + leftOffset;
                if (todayPos < leftOffset) todayPos = leftOffset;
                // predictedPos relativ zu startDate, skaliert auf Bereich [leftOffset, rightOffset]
                const predictedPos = ((wechseldatum - startDate) / (endDate - startDate)) * (rightOffset - leftOffset) + leftOffset;
                // Monatslabels: ebenfalls mit Offset
                const maxLabels = 8;
                const monthsDiff = (endDate.getFullYear() - startDate.getFullYear()) * 12 + (endDate.getMonth() - startDate.getMonth());
                let labelStep = 1;
                if (monthsDiff > maxLabels) {
                  labelStep = Math.ceil(monthsDiff / maxLabels);
                }
                const dateLabels = [];
                let labelCursor = new Date(startDate.getFullYear(), startDate.getMonth(), 1);
                let firstLabelAdded = false;
                while (labelCursor <= endDate) {
                  const isCurrentMonth = labelCursor.getFullYear() === heute.getFullYear() && labelCursor.getMonth() === heute.getMonth();
                  if ((labelCursor.getMonth() - startDate.getMonth()) % labelStep === 0 || isCurrentMonth || !firstLabelAdded) {
                    const pos = ((labelCursor - startDate) / (endDate - startDate)) * (rightOffset - leftOffset) + leftOffset;
                    dateLabels.push({
                      label: labelCursor.toLocaleString('en-US', { month: 'short', year: 'numeric' }),
                      pos
                    });
                    firstLabelAdded = true;
                  }
                  labelCursor.setMonth(labelCursor.getMonth() + 1);
                }
                // Markierter Bereich um Wechseldatum: -20 Tage bis +10 Tage, ebenfalls mit Offset
                const daysSinceStart = Math.round((wechseldatum - startDate) / (1000 * 60 * 60 * 24));
                const contactStart = Math.max(leftOffset, ((daysSinceStart - 20) / totalDays) * (rightOffset - leftOffset) + leftOffset);
                const contactEnd = Math.min(100, ((daysSinceStart + 10) / totalDays) * (rightOffset - leftOffset) + leftOffset);

                // Zeitraum-Objekt finden (jetzt nach confidence)
                const timeRange = getTimeRangeFromDays(adjustedDays);
                const range = zeitraumRanges.find(r => r.label === timeRange) || zeitraumRanges[5];

                // Labels und Farben
                const labels = ["0-3 Monate", "3-6 Monate", "6-9 Monate", "9-12 Monate", "12-18 Monate", "über 18 Monate"];
                const confidences = result.confidence || [];
                const adjustedDaysList = confidences.map(c => c * MAE);
                const topIdx = adjustedDaysList.indexOf(Math.min(...adjustedDaysList)); // Kleinere Tage = früherer Wechsel
                const confidencesCopy = [...adjustedDaysList];
                confidencesCopy[topIdx] = Infinity;
                const secondIdx = confidencesCopy.indexOf(Math.min(...confidencesCopy));
                const secondLabel = labels[secondIdx];
                const secondDays = adjustedDaysList[secondIdx];
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
                          label={`${Math.round(adjustedDays*MAE)} Tagen`}
                          sx={{
                            color: '#000',
                            fontWeight: 700,
                            fontSize: '1.05rem',
                            px: 2,
                            py: 0.5,
                            borderRadius: 2,
                            background: 'transparent'
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
                          {expandedRows.has(index) ? 'Close details' : 'Show details'}
                        </Button>
                      </td>
                    </tr>
                    {expandedRows.has(index) && (
                      <tr>
                        <td colSpan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}>
                          <Box sx={{ borderRadius: '16px', p: '30px', margin: '20px auto', bgcolor: '#fff', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)', maxWidth: '95%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            {/* Zeitstrahl */}
                            <Box sx={{
                              width: '100%',
                              minWidth: 700,
                              overflowX: 'auto',
                              position: 'relative',
                              minHeight: 180,
                              height: 180,
                              px: 4,
                              py: 3,
                              background: '#fff',
                            }}>
                              {/* Zeitstrahl */}
                              <Box sx={{
                                position: 'absolute', top: 70, left: 0, width: '100%',
                                height: 8, borderRadius: 4, bgcolor: '#233038', zIndex: 1
                              }} />
                              {/* Markierter Bereich um Wechseldatum */}
                              <Box sx={{
                                position: 'absolute', top: 62, left: `${contactStart}%`,
                                width: `${contactEnd - contactStart}%`, height: 24,
                                bgcolor: '#07505633', borderRadius: 12, zIndex: 2
                              }} />
                              {/* Marker: Heute (links, farbig, mit Label) */}
                              <Box sx={{
                                position: 'absolute', left: `${todayPos}%`, top: 57,
                                width: 32, height: 32, bgcolor: '#000', borderRadius: '50%',
                                border: '4px solid #fff', zIndex: 4, transform: 'translate(-50%, 0)',
                                boxShadow: '0 0 12px #0008',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                              }} />
                              <span style={{
                                position: 'absolute', left: `${todayPos}%`, top: 100, transform: 'translate(-50%, 0)',
                                color: '#fff', fontWeight: 800, fontSize: 16, background: '#000', borderRadius: 6, padding: '2px 14px', zIndex: 5, boxShadow: '0 2px 8px #0002', letterSpacing: 1
                              }}>today</span>
                              {/* Marker: Wechseldatum (prominent, immer sichtbar) */}
                              <Box sx={{
                                position: 'absolute', left: `${predictedPos}%`, top: 57,
                                width: 32, height: 32, bgcolor: '#075056', borderRadius: '50%',
                                border: '4px solid #fff', zIndex: 4, transform: 'translate(-50%, 0)',
                                boxShadow: '0 0 12px #07505688',
                                display: 'flex', alignItems: 'center', justifyContent: 'center'
                              }} />
                              <span style={{
                                position: 'absolute', left: `${predictedPos}%`, top: 100, transform: 'translate(-50%, 0)',
                                color: '#fff', fontWeight: 800, fontSize: 16, background: '#075056', borderRadius: 6, padding: '2px 14px', zIndex: 5, boxShadow: '0 2px 8px #07505644', letterSpacing: 1
                              }}>job change</span>
                              {/* Monatslabels */}
                              <Box sx={{
                                position: 'absolute', top: 90, width: '100%', display: 'flex', justifyContent: 'space-between'
                              }}>
                                {dateLabels.map((m, i) => (
                                  <span key={i} style={{
                                    position: 'absolute', left: `${m.pos}%`, transform: 'translate(-50%, 0)',
                                    color: '#233038', fontWeight: 600, fontSize: 15,
                                    background: '#fff', borderRadius: 4, padding: '0 8px', zIndex: 4
                                  }}>{m.label}</span>
                                ))}
                              </Box>
                            </Box>
                            {/* Zweitwahrscheinlichkeit */}
                            {secondIdx !== topIdx && (
                              <Typography sx={{ color: secondRange.color, textAlign: 'center', fontSize: '1rem', mt: 5, fontWeight: 500 }}>
                                Zweitwahrscheinlichkeit: {secondLabel} ({Math.round(secondDays)} Tage)
                              </Typography>
                            )}
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
