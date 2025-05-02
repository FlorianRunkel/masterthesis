import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';

const ResultsTable = ({ results, onSave, isSaving }) => {
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
    return (
      <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
        <Box
          sx={{
            p: '30px',
            width: '100%'
          }}
        >
          <Typography variant="h2" sx={{ 
            fontSize: '1.5rem',
            fontWeight: 600,
            color: '#1a1a1a',
            mb: 2
          }}>
            Fehler bei der Verarbeitung
          </Typography>
          <Typography sx={{ color: '#666', mb: 1 }}>{results.error}</Typography>
          <Typography sx={{ color: '#666' }}>{results.message}</Typography>
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

  const successCount = results.filter(r => !r.error).length;
  const errorCount = results.filter(r => r.error).length;

  const toggleDetails = (index) => {
    const newExpandedRows = new Set(expandedRows);
    if (newExpandedRows.has(index)) {
      newExpandedRows.delete(index);
    } else {
      newExpandedRows.add(index);
    }
    setExpandedRows(newExpandedRows);
  };

  const getProbabilityClass = (confidence) => {
    if (confidence < 60) return 'probability-low';
    if (confidence < 80) return 'probability-medium';
    return 'probability-high';
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
            <Typography variant="h2" sx={{ 
              fontSize: '1.5rem',
              fontWeight: 600,
              color: '#1a1a1a',
              mb: 2
            }}>
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
                }}>Wechselwahrscheinlichkeit</th>
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
                const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Nicht angegeben';
                const linkedin = result.linkedinProfile || 'Nicht angegeben';

                if (result.error) {
                  return (
                    <tr key={index} style={{ background: 'rgba(220, 53, 69, 0.05)' }}>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}></td>
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
                        {result.error}
                      </td>
                    </tr>
                  );
                }

                const confidence = result.confidence ? result.confidence[0] * 100 : 0;
                const probabilityClass = getProbabilityClass(confidence);

                return (
                  <React.Fragment key={index}>
                    <tr>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
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
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Link 
                          href={linkedin} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          sx={{
                            color: '#666',
                            textDecoration: 'none',
                            fontSize: '1rem',
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
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                          <Typography sx={{ fontWeight: 600, minWidth: 50, color: probabilityClass === 'probability-low' ? '#dc3545' : probabilityClass === 'probability-medium' ? '#ffc107' : '#28a745' }}>
                            {confidence.toFixed(0)}%
                          </Typography>
                          <Box sx={{ flexGrow: 1, height: 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
                            <Box
                              sx={{
                                height: '100%',
                                width: `${confidence}%`,
                                bgcolor: probabilityClass === 'probability-low' ? '#dc3545' :
                                        probabilityClass === 'probability-medium' ? '#ffc107' : '#28a745',
                                borderRadius: 1,
                                transition: 'width 0.3s ease'
                              }}
                            />
                          </Box>
                        </Box>
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Button
                          onClick={() => toggleDetails(index)}
                          sx={{
                            bgcolor: '#001B41',
                            color: 'white',
                            textTransform: 'none',
                            px: 2,
                            py: 1,
                            borderRadius: '8px',
                            fontSize: '0.8rem',
                            fontWeight: 600,
                            '&:hover': {
                              bgcolor: '#FF5F00'
                            }
                          }}
                        >
                          {expandedRows.has(index) ? 'Hide Details' : 'Show Details'}
                        </Button>
                      </td>
                    </tr>
                    {expandedRows.has(index) && (
                      <tr>
                        <td colspan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}>
                          <Box sx={{ 
                            borderRadius: '16px',
                            p: '30px',
                            margin: '20px auto',
                            bgcolor: '#fff',
                            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.2)',
                            maxWidth: '95%'
                          }}>

                            <Box sx={{ color: '#666' }}>
                              {(() => {
                                // Sortiere und filtere die wichtigsten Features
                                const explanations = (result.explanations || [])
                                  .sort((a, b) => parseFloat(b.impact_percentage) - parseFloat(a.impact_percentage));
                                const top3 = explanations.slice(0, 3);
                                const sonstigeSumme = explanations.slice(3).reduce((sum, f) => sum + parseFloat(f.impact_percentage), 0);

                                // Farben wie in PredictionResult.jsx
                                const getBarColors = [
                                  '#28a745', // grün
                                  '#ffc107', // gelb
                                  '#dc3545', // rot
                                  '#b0b0b0'  // grau für Sonstiges
                                ];

                                // Daten für den gestapelten Balken
                                const barData = [
                                  ...top3.map((f, i) => ({
                                    ...f,
                                    color: getBarColors[i]
                                  })),
                                  ...(sonstigeSumme > 0 ? [{
                                    feature: 'Sonstiges',
                                    impact_percentage: sonstigeSumme,
                                    color: getBarColors[3]
                                  }] : [])
                                ];

                                if (barData.length === 0) {
                                  return (
                                    <Typography sx={{ color: '#666' }}>
                                      Keine Feature-Erklärungen verfügbar.
                                    </Typography>
                                  );
                                }

                                return (
                                  <Box>
                                    {/* Überschrift für die Erklärung */}
                                    <Typography sx={{ fontWeight: 600, fontSize: '1.15rem', color: '#001B41', mb: 2 }}>
                                      Vorhersage-Erklärung
                                    </Typography>
                                    {/* Gestapelter Balken */}
                                    <Box sx={{ display: 'flex', width: '100%', height: 28, borderRadius: 2, overflow: 'hidden', boxShadow: 1, mb: 2 }}>
                                      {barData.map((item, idx) => (
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
                                            fontSize: '0.95rem',
                                            borderRight: idx < barData.length - 1 ? '2px solid #fff' : 'none',
                                            transition: 'width 0.3s ease'
                                          }}
                                        >
                                          {item.impact_percentage > 8 && `${item.impact_percentage.toFixed(1)}%`}
                                        </Box>
                                      ))}
                                    </Box>
                                    {/* Legende */}
                                    <Box sx={{ display: 'flex', gap: 2, mt: 1, flexWrap: 'wrap' }}>
                                      {barData.map(item => (
                                        <Box key={item.feature} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                          <Box sx={{ width: 16, height: 16, bgcolor: item.color, borderRadius: 1, mr: 0.5 }} />
                                          <Typography variant="body2">{item.feature}</Typography>
                                        </Box>
                                      ))}
                                    </Box>
                                  </Box>
                                );
                              })()}
                            </Box>
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

export default ResultsTable; 