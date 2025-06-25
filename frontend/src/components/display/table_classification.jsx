import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link, useMediaQuery, useTheme, IconButton } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import PredictionResultClassification from '../prediction/prediction_classification';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

// ResultsTableClassification: Displays a table of classification results for candidates
const ResultsTableClassification = ({ results, onSave, isSaving, originalProfiles }) => {
  // --- State Management ---
  // Set of selected candidates for saving
  const [selectedCandidates, setSelectedCandidates] = useState(new Set());
  // Set of expanded rows for details view
  const [expandedRows, setExpandedRows] = useState(new Set());
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // --- Early Return: No results ---
  if (!results) return null;

  // --- Statistics ---
  const successCount = results.filter(r => !r.error).length;
  const errorCount = results.filter(r => r.error).length;

  // --- Candidate Selection Handler ---
  const handleSelectCandidate = (index) => {
    const newSelected = new Set(selectedCandidates);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedCandidates(newSelected);
  };

  // --- Save Selected Candidates Handler ---
  const handleSaveSelected = () => {
    const candidatesToSave = Array.from(selectedCandidates).map(index => ({
      ...(originalProfiles && originalProfiles[index] ? originalProfiles[index] : {}),
      ...results[index],
      savedAt: new Date().toISOString()
    }));
    onSave(candidatesToSave);
  };

  // --- Row Expand/Collapse Handler ---
  const toggleDetails = (index) => {
    const newExpandedRows = new Set(expandedRows);
    if (newExpandedRows.has(index)) {
      newExpandedRows.delete(index);
    } else {
      newExpandedRows.add(index);
    }
    setExpandedRows(newExpandedRows);
  };

  // --- Helper: Get Probability Class for Styling ---
  const getProbabilityClass = (confidence) => {
    if (confidence < 40) return 'probability-low';
    if (confidence < 70) return 'probability-medium';
    return 'probability-high';
  };

  // --- Helper: Get Status Icon by Probability ---
  const getStatusIcon = (probabilityClass) => {
    switch (probabilityClass) {
      case 'probability-high':
        return (<CheckCircleIcon sx={{ color: '#4CAF50', fontSize: 22, verticalAlign: 'middle' }} />);
      default:
        return (<CancelIcon sx={{ color: '#F44336', fontSize: 22, verticalAlign: 'middle' }} />);
    }
  };

  // --- Helper: Get Color by Probability ---
  const getColorByProbability = (probabilityClass) => {
    switch (probabilityClass) {
      case 'probability-low':
        return '#d81b3b';
      case 'probability-medium':
        return '#FFC03D';
      default:
        return '#2e6f40';
    }
  };

  // --- Error Handling: Show Error Message if Results Contain Error ---
  if (results.error) {
    return (
      <Box sx={{ maxWidth: '1200px', ml: 0 }}>
        <Box sx={{ p: '30px', width: '100%' }}>
          <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#1a1a1a', mb: 2 }}>Error during processing</Typography>
          <Typography sx={{ color: '#666', mb: 1 }}>{results.error}</Typography>
          <Typography sx={{ color: '#666' }}>{results.message}</Typography>
          {results.requirements && (
            <>
              <Typography sx={{ mt: 2, mb: 1, color: '#666' }}>Required columns:</Typography>
              <ul style={{ color: '#666', margin: 0, paddingLeft: 20 }}>{results.requirements.map((req, index) => (<li key={index}>{req}</li>))}</ul>
            </>
          )}
        </Box>
      </Box>
    );
  }

  // --- Main Render: Table of Results ---
  return (
    <Box sx={{margin: '0 auto' }}>
      {/* --- Table Header: Summary and Save Button --- */}
      <Box sx={{ bgcolor: '#fff', borderRadius: '13px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 3.2, width: '100%' }}>
        <Box sx={{ p: '24px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Typography variant="h2" sx={{ fontSize: '1.3rem', fontWeight: 800, color: '#1a1a1a', mb: 1.6 }}>Summary of batch processing</Typography>
            <Typography sx={{ mb: 0.8, color: '#666', fontSize: '0.88rem' }}><strong>Successfully processed:</strong> {successCount} candidates</Typography>
            <Typography sx={{ color: '#666', fontSize: '0.88rem' }}><strong>Error:</strong> {errorCount} candidates</Typography>
          </div>
          {selectedCandidates.size > 0 && (
            <Button variant="contained" color="primary" onClick={handleSaveSelected} disabled={isSaving} startIcon={isSaving ? <CircularProgress size={19} sx={{ color: 'white' }} /> : <SaveIcon />} sx={{ bgcolor: '#001242', color: 'white', p: '8px 16px', borderRadius: '6.4px', textTransform: 'none', fontWeight: 600, fontSize: '0.8rem', '&:hover': { bgcolor: '#EB7836' } }}>{isSaving ? 'Save...' : `${selectedCandidates.size} candidates to save`}</Button>
          )}
        </Box>
        {/* --- Table Content: Mobile and Desktop Views --- */}
        <Box sx={{ overflowX: 'auto', width: '100%' }}>
          {isMobile ? (
            // --- Mobile View: Render as List ---
            results.map((result, index) => {
              const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Not specified';
              const linkedin = result.linkedinProfile || 'Not specified';
              const confidence = result.confidence ? result.confidence[0] * 100 : 0;
              const probabilityClass = getProbabilityClass(confidence);
              const statusIcon = getStatusIcon(probabilityClass);
              const isExpanded = expandedRows.has(index);
              return (
                <Box key={index} sx={{ mb: 0, p: 2, borderBottom: '1px solid #eee', '&:last-child': { borderBottom: 'none' } }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{name}</Typography>
                      <Typography sx={{ fontSize: '0.85rem', color: '#888', mt: 0.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}><Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#888', textDecoration: 'none' }}>{linkedin}</Link></Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>{statusIcon}<IconButton size="small" onClick={() => toggleDetails(index)} sx={{ ml: 1 }} aria-label={isExpanded ? 'Collapse details' : 'Expand details'}>{isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}</IconButton></Box>
                  </Box>
                  {isExpanded && (<Box sx={{ mt: 2, width: '100%' }}><PredictionResultClassification prediction={result} /></Box>)}
                </Box>
              );
            })
          ) : (
            // --- Desktop View: Render as Table ---
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem', width: '32px' }}></th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>Name</th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 600, fontSize: '0.88rem' }}>LinkedIn</th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'center', fontWeight: 900, fontSize: '0.88rem' }}>Propensity to Change</th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>Explanation</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result, index) => {
                  const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Not specified';
                  const linkedin = result.linkedinProfile || 'Not specified';
                  const confidence = result.confidence ? result.confidence[0] * 100 : 0;
                  const probabilityClass = getProbabilityClass(confidence);
                  const color = getColorByProbability(probabilityClass);
                  const isExpanded = expandedRows.has(index);
                  // --- Error Row ---
                  if (result.error) {
                    return (
                      <tr key={index} style={{ background: 'rgba(220, 53, 69, 0.05)' }}>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee' }}></td>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee' }}>{name}</td>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee' }}><Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#666', textDecoration: 'none', fontSize: '0.85rem', opacity: 0.8, transition: 'opacity 0.2s ease', '&:hover': { opacity: 1 } }}>{linkedin}</Link></td>
                        <td colSpan="2" style={{ padding: '12px 24px', borderBottom: '1px solid #eee', color: '#FF2525' }}>{result.error}</td>
                      </tr>
                    );
                  }
                  // --- Normal Row ---
                  return (
                    <React.Fragment key={index}>
                      <tr>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee' }}><Checkbox checked={selectedCandidates.has(index)} onChange={() => handleSelectCandidate(index)} sx={{ color: '#666', '&.Mui-checked': { color: '#EB7836' } }} /></td>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>{name}</td>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}><Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#666', textDecoration: 'none', fontSize: '0.88rem', opacity: 0.8, transition: 'opacity 0.2s ease', '&:hover': { opacity: 1 } }}>{linkedin}</Link></td>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}><Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%' }}><span style={{ color, fontSize: '0.88rem', fontWeight: 800 }}>{`${Math.round(confidence)}%`}</span></Box></td>
                        <td style={{ padding: '12px 24px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>
                          <Button
                            onClick={() => toggleDetails(index)}
                            sx={{
                              bgcolor: isExpanded ? '#EB7836' : '#001242',
                              color: 'white',
                              textTransform: 'none',
                              px: 0,
                              py: 0,
                              borderRadius: '8px',
                              fontSize: '0.88rem',
                              fontWeight: 600,
                              width: 120,
                              height: 40,
                              minWidth: 120,
                              minHeight: 40,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              gap: 1,
                              '&:hover': { bgcolor: '#EB7836' }
                            }}
                          >
                            {isExpanded ? 'Collapse' : 'Expand'}
                            {isExpanded ? <ExpandLessIcon sx={{ fontSize: 20, ml: 1 }} /> : <ExpandMoreIcon sx={{ fontSize: 20, ml: 1 }} />}
                          </Button>
                        </td>
                      </tr>
                      {isExpanded && (<tr><td colSpan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}><Box sx={{ margin: '0px auto', p: 3, maxWidth: '100%' }}><PredictionResultClassification prediction={result} /></Box></td></tr>)}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default ResultsTableClassification;