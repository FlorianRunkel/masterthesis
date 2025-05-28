import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import PredictionResultClassification from '../prediction/PredictionResultClassification';

const ResultsTableClassification = ({ results, onSave, isSaving, originalProfiles }) => {
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
      ...(originalProfiles && originalProfiles[index] ? originalProfiles[index] : {}),
      ...results[index],
      savedAt: new Date().toISOString()
    }));
    onSave(candidatesToSave);
  };

  if (results.error) {
    return (
      <Box sx={{ maxWidth: '1200px', ml: 0 }}>
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
            Error during processing
          </Typography>
          <Typography sx={{ color: '#666', mb: 1 }}>{results.error}</Typography>
          <Typography sx={{ color: '#666' }}>{results.message}</Typography>
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
    if (confidence < 85) return 'probability-medium';
    return 'probability-high';
  };

  return (
    <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
      <Box sx={{ bgcolor: '#fff', borderRadius: '13px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 3.2, width: '100%' }}>
        <Box sx={{ p: '24px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Typography variant="h2" sx={{ fontSize: '1.3rem', fontWeight: 800, color: '#1a1a1a', mb: 1.6 }}>Summary of batch processing</Typography>
            <Typography sx={{ mb: 0.8, color: '#666' , fontSize: '0.88rem'}}><strong>Successfully processed:</strong> {successCount} candidates</Typography>
            <Typography sx={{ color: '#666' , fontSize: '0.88rem'}}><strong>Error:</strong> {errorCount} candidates</Typography>
          </div>
          {selectedCandidates.size > 0 && (
            <Button
              variant="contained"
              color="primary"
              onClick={handleSaveSelected}
              disabled={isSaving}
              startIcon={isSaving ? <CircularProgress size={19} sx={{ color: 'white' }} /> : <SaveIcon />}
              sx={{ bgcolor: '#13213C', color: 'white', p: '8px 16px', borderRadius: '6.4px', textTransform: 'none', fontWeight: 600, fontSize: '0.8rem', '&:hover': { bgcolor: '#FF8000' } }}
            >
              {isSaving ? 'Save...' : selectedCandidates.size + ' candidates to save'}
            </Button>
          )}
        </Box>

        <Box sx={{ overflowX: 'auto', width: '100%' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ background: '#13213C', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem', width: '32px' }}></th>
                <th style={{ background: '#13213C', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>Name</th>
                <th style={{ background: '#13213C', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 600, fontSize: '0.88rem' }}>LinkedIn</th>
                <th style={{ background: '#13213C', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>Change Probability</th>
                <th style={{ background: '#13213C', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}></th>
              </tr>
            </thead>
            <tbody>
              {results.map((result, index) => {
                const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Nicht angegeben';
                const linkedin = result.linkedinProfile || 'Nicht angegeben';

                if (result.error) {
                  return (
                    <tr key={index} style={{ background: 'rgba(220, 53, 69, 0.05)' }}>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee' }}></td>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee' }}>{name}</td>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee' }}>
                        <Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#666', textDecoration: 'none', fontSize: '0.85rem', opacity: 0.8, transition: 'opacity 0.2s ease', '&:hover': { opacity: 1 } }}>{linkedin}</Link>
                      </td>
                      <td colSpan="2" style={{ padding: '12px 24px', borderBottom: '1px solid #eee', color: '#FF2525' }}>{result.error}</td>
                    </tr>
                  );
                }

                const confidence = result.confidence ? result.confidence[0] * 100 : 0;
                const probabilityClass = getProbabilityClass(confidence);

                return (
                  <React.Fragment key={index}>
                    <tr>
                      <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee' }}>
                        <Checkbox checked={selectedCandidates.has(index)} onChange={() => handleSelectCandidate(index)} sx={{ color: '#666', '&.Mui-checked': { color: '#FF8000' } }} />
                      </td>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>{name}</td>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>
                        <Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#666', textDecoration: 'none', fontSize: '0.88rem', opacity: 0.8, transition: 'opacity 0.2s ease', '&:hover': { opacity: 1 } }}>{linkedin}</Link>
                      </td>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                          <Typography sx={{ fontWeight: 600, minWidth: 50, color: probabilityClass === 'probability-low' ? '#FF2525' : probabilityClass === 'probability-medium' ? '#FFC03D' : '#8AD265' , fontSize: '0.88rem'}}>{confidence.toFixed(0)}%</Typography>
                          <Box sx={{ flexGrow: 1, height: 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
                            <Box sx={{ height: '100%', width: `${confidence}%`, bgcolor: probabilityClass === 'probability-low' ? '#FF2525' : probabilityClass === 'probability-medium' ? '#FFC03D' : '#8AD265', borderRadius: 1, transition: 'width 0.3s ease' }} />
                          </Box>
                        </Box>
                      </td>
                      <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>
                        <Button onClick={() => toggleDetails(index)} sx={{ bgcolor: '#13213C', color: 'white', textTransform: 'none', px: 2, py: 1, borderRadius: '8px', fontSize: '0.8rem', fontWeight: 600, '&:hover': { bgcolor: '#FF8000' } }}>
                          {expandedRows.has(index) ? 'Hide Details' : 'Show Details'}
                        </Button>
                      </td>
                    </tr>
                    {expandedRows.has(index) && (
                      <tr>
                        <td colSpan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}>
                          <Box sx={{margin: '0px auto', p: 3, maxWidth: '100%' }}>
                            <PredictionResultClassification prediction={result} />
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

export default ResultsTableClassification; 