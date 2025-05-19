import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link, Chip } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import Timeline from './Timeline';

const ResultsTableTimeSeries = ({ results, onSave, isSaving, originalProfiles }) => {
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
      ...originalProfiles[index],
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
      <Box sx={{ bgcolor: '#fff', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4, width: '100%' }}>
        <Box sx={{ p: '30px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#1a1a1a', mb: 2 }}>Batch Processing Summary</Typography>
            <Typography sx={{ mb: 1, color: '#666' }}><strong>Successfully processed:</strong> {successCount} candidates</Typography>
            <Typography sx={{ color: '#666' }}><strong>Errors:</strong> {errorCount} candidates</Typography>
          </div>
          {selectedCandidates.size > 0 && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleSaveSelected}
              disabled={isSaving}
              startIcon={isSaving ? <CircularProgress size={24} sx={{ color: 'white' }} /> : <SaveIcon />}
              sx={{ bgcolor: '#13213C', color: 'white', p: '10px 20px', borderRadius: '8px', textTransform: 'none', fontWeight: 600, '&:hover': { bgcolor: '#FF8000' } }}
            >
              {isSaving ? 'Saving...' : selectedCandidates.size + ' Save candidates'}
            </Button>
          )}
        </Box>

        <Box sx={{ overflowX: 'auto', width: '100%' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ background: '#13213C', color: 'white', padding: '15px 30px', textAlign: 'left', fontWeight: 900, fontSize: '1.1rem', width: '40px' }}></th>
                <th style={{ background: '#13213C', color: 'white', padding: '15px 30px', textAlign: 'left', fontWeight: 900, fontSize: '1.1rem' }}>Name</th>
                <th style={{ background: '#13213C', color: 'white', padding: '15px 30px', textAlign: 'left', fontWeight: 900, fontSize: '1.1rem' }}>LinkedIn</th>
                <th style={{ background: '#13213C', color: 'white', padding: '15px 30px', textAlign: 'left', fontWeight: 900, fontSize: '1.1rem' }}>Job change period</th>
                <th style={{ background: '#13213C', color: 'white', padding: '15px 30px', textAlign: 'left', fontWeight: 900, fontSize: '1.1rem' }}></th>
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
                        <Chip label={result.confidence} sx={{ bgcolor: 'grey', color: '#fff', fontWeight: 700, fontSize: '1.05rem', px: 2, py: 0.5, borderRadius: 2 }} />
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#666', textDecoration: 'none', fontSize: '0.85rem', opacity: 0.8, transition: 'opacity 0.2s ease', '&:hover': { opacity: 1 } }}>{linkedin}</Link>
                      </td>
                      <td colSpan="2" style={{ padding: '15px 30px', borderBottom: '1px solid #eee', color: '#FF2525' }}>{result.error || 'Processing error'}</td>
                    </tr>
                  );
                }
                // Tage werden direkt aus der Confidence genommen

                return (
                  <React.Fragment key={index}>
                    <tr style={{ transition: 'background 0.2s', cursor: 'pointer', ':hover': { background: '#f5f8ff' } }}>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee', textAlign: 'center' }}>
                        <Checkbox checked={selectedCandidates.has(index)} onChange={() => handleSelectCandidate(index)} sx={{ color: '#666', '&.Mui-checked': { color: '#666' } }} />
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee', fontWeight: 500, fontSize: '1.08rem' }}>{name}</td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#13213C', fontWeight: 500, fontSize: '1rem', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 1, '&:hover': { color: '#FF8000', textDecoration: 'underline' } }}>{linkedin.replace(/^https?:\/\/|^www\./, '')}</Link>
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee' }}>
                        <Chip label={`${result.confidence} Tag${result.confidence === 1 ? '' : 'e'}`} sx={{ color: '#000', fontWeight: 700, fontSize: '1.05rem', px: 2, py: 0.5, borderRadius: 2, background: 'transparent' }} />
                      </td>
                      <td style={{ padding: '15px 30px', borderBottom: '1px solid #eee', textAlign: 'center' }}>
                        <Button onClick={() => toggleDetails(index)} sx={{ bgcolor: '#13213C', color: 'white', textTransform: 'none', px: 2.5, py: 1.2, borderRadius: '8px', fontSize: '1rem', fontWeight: 600, minWidth: 0, '&:hover': { bgcolor: '#FF8000' }, display: 'flex', alignItems: 'center', gap: 1 }} endIcon={expandedRows.has(index) ? <ExpandLessIcon /> : <ExpandMoreIcon />}>
                          {expandedRows.has(index) ? 'Close details' : 'Show details'}
                        </Button>
                      </td>
                    </tr>
                    {expandedRows.has(index) && (
                      <tr>
                        <td colSpan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}>
                          <Box sx={{ borderRadius: '16px', p: '20px', margin: '20px auto', bgcolor: '#fff', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)', maxWidth: '95%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                            <Timeline prediction={result} />
                            {result.llm_explanation && (
                              <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}>
                                <Typography sx={{ color: '#444', fontSize: '1.1rem', lineHeight: 1.9 }}>{result.llm_explanation}</Typography>
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

export default ResultsTableTimeSeries;
