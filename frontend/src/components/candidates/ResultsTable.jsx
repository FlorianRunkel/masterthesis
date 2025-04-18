import React, { useState } from 'react';
import { Box, Typography, Button } from '@mui/material';

const ResultsTable = ({ results }) => {
  const [expandedRows, setExpandedRows] = useState(new Set());

  if (!results) return null;

  if (results.error) {
    return (
      <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
        <Box
          sx={{
            bgcolor: '#fff',
            borderRadius: '16px',
            p: '30px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            mb: 4,
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
    if (confidence < 50) return 'probability-low';
    if (confidence < 75) return 'probability-medium';
    return 'probability-high';
  };

  const getProbabilityText = (confidence) => {
    if (confidence < 50) return 'Niedrig';
    if (confidence < 75) return 'Mittel';
    return 'Hoch';
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
        <Box sx={{ p: '30px', borderBottom: '1px solid #eee' }}>
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
                  fontWeight: 600,
                  fontSize: '0.9rem'
                }}>Name</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 600,
                  fontSize: '0.9rem'
                }}>LinkedIn</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 600,
                  fontSize: '0.9rem'
                }}>Wechselwahrscheinlichkeit</th>
                <th style={{
                  background: '#001B41',
                  color: 'white',
                  padding: '15px 30px',
                  textAlign: 'left',
                  fontWeight: 600,
                  fontSize: '0.9rem'
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
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <a href={linkedin} target="_blank" rel="noopener noreferrer" style={{ color: '#666', textDecoration: 'none' }}>
                          {linkedin}
                        </a>
                      </td>
                      <td colSpan="2" style={{ padding: '15px 30px', borderBottom: '1px solid #eee', color: '#dc3545' }}>
                        {result.error}
                      </td>
                    </tr>
                  );
                }

                const confidence = result.confidence ? result.confidence[0] * 100 : 0;
                const probabilityClass = getProbabilityClass(confidence);
                const probabilityText = getProbabilityText(confidence);

                return (
                  <React.Fragment key={index}>
                    <tr>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <a href={linkedin} target="_blank" rel="noopener noreferrer" style={{ color: '#333', textDecoration: 'none' }}>
                          {linkedin}
                        </a>
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
                            fontSize: '0.9rem',
                            fontWeight: 600,
                            '&:hover': {
                              bgcolor: '#FF5F00'
                            }
                          }}
                        >
                          {expandedRows.has(index) ? 'Details ausblenden' : 'Details anzeigen'}
                        </Button>
                      </td>
                    </tr>
                    {expandedRows.has(index) && (
                      <tr>
                        <td colSpan="4" style={{ padding: '30px', background: '#F5F5F7' }}>
                          <Box sx={{ 
                            display: 'grid', 
                            gridTemplateColumns: '1fr 1fr', 
                            gap: 3,
                            maxWidth: '1200px',
                            margin: '0 auto'
                          }}>
                            <Box sx={{ 
                              bgcolor: '#fff', 
                              p: '30px', 
                              borderRadius: '16px', 
                              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                              width: '100%'
                            }}>
                              <Typography sx={{ 
                                fontSize: '1.1rem',
                                fontWeight: 600,
                                color: '#1a1a1a',
                                mb: 2
                              }}>
                                Verwendete Features:
                              </Typography>
                              <ul style={{ margin: 0, paddingLeft: 20, color: '#666' }}>
                                {Object.entries(result.features || {}).map(([key, value], i) => (
                                  <li key={i} style={{ marginBottom: '8px' }}>
                                    <strong>{key}:</strong> {value.toFixed(3)}
                                  </li>
                                ))}
                              </ul>
                            </Box>
                            <Box sx={{ 
                              bgcolor: '#fff', 
                              p: '30px', 
                              borderRadius: '16px', 
                              boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                              width: '100%'
                            }}>
                              <Typography sx={{ 
                                fontSize: '1.1rem',
                                fontWeight: 600,
                                color: '#1a1a1a',
                                mb: 2
                              }}>
                                Empfehlungen:
                              </Typography>
                              <ul style={{ margin: 0, paddingLeft: 20, color: '#666' }}>
                                {(result.recommendations || []).map((rec, i) => (
                                  <li key={i} style={{ marginBottom: '8px' }}>{rec}</li>
                                ))}
                              </ul>
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