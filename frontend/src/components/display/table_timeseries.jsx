import React, { useState } from 'react';
import { Box, Typography, Button, Checkbox, CircularProgress, Link, Chip, IconButton, useMediaQuery, useTheme, FormControl, Select, MenuItem, InputLabel, Menu } from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import FilterListIcon from '@mui/icons-material/FilterList';
import Timeline from '../prediction/helper_timeline';

const ResultsTableTimeSeries = ({ results, onSave, isSaving, originalProfiles }) => {

  const [selectedCandidates, setSelectedCandidates] = useState(new Set());
  const [expandedRows, setExpandedRows] = useState(new Set());
  const [timeFilter, setTimeFilter] = useState('all');
  const [filterAnchorEl, setFilterAnchorEl] = useState(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  if (!results) return null;

  const successCount = results.filter(r => !r.error && r.status !== 'error').length;
  const errorCount = results.filter(r => r.error || r.status === 'error').length;

  // Filter results based on time filter
  const getFilteredResults = () => {
    if (timeFilter === 'all') return results;
    
    return results.filter(result => {
      if (result.error || result.status === 'error') return true; // Always show errors
      
      const months = result.confidence / 30.44;

      switch (timeFilter) {
        case 'short':
          return months < 12; // Under 12 months
        case 'medium':
          return months >= 12 && months < 24; // 12 months to 2 years
        case 'long':
          return months >= 24; // Over 2 years
        default:
          return true;
      }
    });
  };

  const filteredResults = getFilteredResults();

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

  const handleFilterClick = (event) => {
    setFilterAnchorEl(event.currentTarget);
  };

  const handleFilterClose = () => {
    setFilterAnchorEl(null);
  };

  const handleFilterSelect = (filterValue) => {
    setTimeFilter(filterValue);
    setFilterAnchorEl(null);
  };

  const toggleDetails = (index) => {
    const newExpandedRows = new Set(expandedRows);
    if (newExpandedRows.has(index)) {
      newExpandedRows.delete(index);
    } else {
      newExpandedRows.add(index);
    }
    setExpandedRows(newExpandedRows);
  };

  const formatJobChangePeriod = (confidence) => {
    if (!confidence) return { text: 'N/A', color: '#666' };
    const months = confidence / 30.44;
    const years = Math.floor(months / 12);
    const remainingMonths = Math.floor(months % 12);

    let startMonth, endMonth;

    if (remainingMonths <= 1) {
      startMonth = 0;
      endMonth = 2;
    } else if (remainingMonths >= 10) {
      startMonth = 9;
      endMonth = 11;
    } else {
      startMonth = Math.max(0, remainingMonths - 1);
      endMonth = Math.min(11, remainingMonths + 1);
    }

    let text;
    if (years === 0) {
      if (startMonth === 0) {
        text = `${endMonth} months`;
      } else {
        text = `${startMonth}-${endMonth} months`;
      }
    } else {
      const yearLabel = `${years} year${years === 1 ? '' : 's'}`;
      if (startMonth === 0) {
        text = `${yearLabel} ${endMonth} months`;
      } else {
        text = `${yearLabel} ${startMonth}-${endMonth} months`;
      }
    }

    let color;
    if (months < 12) {
      color = '#2e6f40';
    } else if (months < 24) {
      color = '#FFC03D';
    } else {
      color = '#d81b3b';
    }
    return { text, color };
  }

  if (results.error) {
    const errorMessage = typeof results.error === 'string' && results.error.includes('confidence')
      ? 'Processing error: Recommendations could not be generated.'
      : results.error;
    return (
      <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
        <Box sx={{ p: '30px', width: '100%' }}>
          <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#1a1a1a', mb: 2 }}>Processing error</Typography>
          <Typography sx={{ color: '#666', mb: 1 }}>{errorMessage}</Typography>
          {results.message && (<Typography sx={{ color: '#666' }}>{results.message}</Typography>)}
          {results.requirements && (<><Typography sx={{ mt: 2, mb: 1, color: '#666' }}>Required columns:</Typography><ul style={{ color: '#666', margin: 0, paddingLeft: 20 }}>{results.requirements.map((req, index) => (<li key={index}>{req}</li>))}</ul></>)}
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ml: 0 }}>
      <Box sx={{ bgcolor: '#fff', borderRadius: '13px', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 3.2, width: '100%' }}>
        <Box sx={{ p: '24px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Typography variant="h2" sx={{ fontSize: '1.3rem', fontWeight: 800, color: '#1a1a1a', mb: 1.6 }}>Batch Processing Summary</Typography>
            <Typography sx={{ mb: 0.8, color: '#666', fontSize: '0.88rem' }}><strong>Successfully processed:</strong> {successCount} candidates</Typography>
            <Typography sx={{ color: '#666', fontSize: '0.88rem' }}><strong>Errors:</strong> {errorCount} candidates</Typography>
          </div>
          {selectedCandidates.size > 0 && (
            <Button variant="contained" color="primary" onClick={handleSaveSelected} disabled={isSaving} startIcon={isSaving ? <CircularProgress size={19} sx={{ color: 'white' }} /> : <SaveIcon />} sx={{ bgcolor: '#001242', color: 'white', p: isMobile ? '6px 12px' : '8px 16px', borderRadius: '6.4px', textTransform: 'none', fontWeight: 600, fontSize: isMobile ? '0.75rem' : '0.8rem','&:hover': { bgcolor: '#EB7836' }, display: 'flex', alignItems: 'center', gap: 0.5,  whiteSpace: 'nowrap'}}>{isSaving ? 'Saving...' : `${selectedCandidates.size} ${isMobile ? 'Save' : 'Save candidates'}`}</Button>
          )}
        </Box>
        <Box sx={{ overflowX: 'auto', width: '100%' }}>
          {isMobile ? (
            filteredResults.map((result, index) => {
              const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Not specified';
              const linkedin = result.linkedinProfile || 'Not specified';
              const jobChangePeriod = formatJobChangePeriod(result.confidence);
              const isExpanded = expandedRows.has(index);
              return (
                <Box key={index} sx={{ mb: 0, p: 2, borderBottom: '1px solid #eee', '&:last-child': { borderBottom: 'none' } }}>
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 1 }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: '#001242' }}>{name}</Typography>
                      <Typography sx={{ fontSize: '0.85rem', color: '#888', mt: 0.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}><Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#888', textDecoration: 'none', '&:hover': { color: '#EB7836' } }}>{linkedin}</Link></Typography>
                      <Typography sx={{ fontSize: '0.95rem', color: jobChangePeriod.color, mt: 0.5, fontWeight: 600, textAlign: 'center' }}>Job Change Period: {jobChangePeriod.text}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0 }}>
                      <Checkbox checked={selectedCandidates.has(index)} onChange={() => handleSelectCandidate(index)} sx={{ color: '#666', '&.Mui-checked': { color: '#EB7836' }, p: 0.5 }} />
                      <IconButton size="small" onClick={() => toggleDetails(index)} sx={{ p: 0.5, color: '#001242', '&:hover': { color: '#EB7836' } }} aria-label={isExpanded ? 'Collapse' : 'Expand'}>{isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}</IconButton>
                    </Box>
                  </Box>
                  {isExpanded && (<Box sx={{ mt: 1, width: '100%', borderTop: '1px solid #eee', pt: 2 }}><Timeline prediction={result} />{result.llm_explanation && (<Box sx={{ mt: 3, p: 1, bgcolor: '#f5f5f5', borderRadius: 2 }}><Typography sx={{ color: '#444', fontSize: '0.88rem', lineHeight: 1.9 }}>{result.llm_explanation}</Typography></Box>)}</Box>)}
                </Box>
              );
            })
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem', width: '32px' }}></th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>Name</th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>LinkedIn</th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'center', fontWeight: 900, fontSize: '0.88rem', position: 'relative' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer'}} onClick={handleFilterClick}>
                      <span>Job change period</span>
                      <FilterListIcon sx={{ fontSize: '1rem', ml: 1, color: 'white' }} />
                    </Box>
                  </th>
                  <th style={{ background: '#001242', color: 'white', padding: '12px 24px', textAlign: 'left', fontWeight: 900, fontSize: '0.88rem' }}>Explanation</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map((result, index) => {
                  const name = `${result.firstName || ''} ${result.lastName || ''}`.trim() || 'Not specified';
                  const linkedin = result.linkedinProfile || 'Not specified';
                  const isExpanded = expandedRows.has(index);
                  if (result.error || result.status === 'error') {
                    return (
                      <tr key={index} style={{ background: 'rgba(220, 53, 69, 0.05)' }}>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}></td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}><Chip label={result.confidence} sx={{ bgcolor: 'grey', color: '#fff', fontWeight: 700, fontSize: '0.85rem', px: 2, py: 0.5, borderRadius: 2 }} /></td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}>{name}</td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', fontSize: '0.88rem' }}><Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#666', textDecoration: 'none', fontSize: '0.75rem', opacity: 0.8, transition: 'opacity 0.2s ease', '&:hover': { opacity: 1 } }}>{linkedin}</Link></td>
                        <td colSpan="2" style={{ padding: '10px 22px', borderBottom: '1px solid #eee', color: '#FF2525', fontSize: '0.88rem' }}>{result.error || 'Processing error'}</td>
                      </tr>
                    );
                  }
                  return (
                    <React.Fragment key={index}>
                      <tr style={{ transition: 'background 0.2s', cursor: 'pointer', ':hover': { background: '#f5f8ff' } }}>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', textAlign: 'center' }}><Checkbox checked={selectedCandidates.has(index)} onChange={() => handleSelectCandidate(index)} sx={{ color: '#666', '&.Mui-checked': { color: '#EB7836' }, width: '10px', height: '10px' }} /></td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', fontWeight: 500, fontSize: '0.88rem' }}>{name}</td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee' }}><Link href={linkedin} target="_blank" rel="noopener noreferrer" sx={{ color: '#001242', fontWeight: 500, fontSize: '0.88rem', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 1, '&:hover': { color: '#EB7836', textDecoration: 'underline' } }}>{linkedin.replace(/^https?:\/\/|^www\./, '')}</Link></td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', textAlign: 'center' }}><Chip label={formatJobChangePeriod(result.confidence).text} sx={{ color: formatJobChangePeriod(result.confidence).color, fontWeight: 700, fontSize: '0.88rem', px: 2, py: 0.5, borderRadius: 2, background: 'transparent' }} /></td>
                        <td style={{ padding: '10px 22px', borderBottom: '1px solid #eee', textAlign: 'center' }}><Button onClick={() => toggleDetails(index)} sx={{ bgcolor: '#001242', color: 'white', textTransform: 'none', px: 2, py: 1, borderRadius: '6.4px', fontSize: '0.88rem', fontWeight: 600, minWidth: 0, '&:hover': { bgcolor: '#EB7836' }, display: 'flex', alignItems: 'center', gap: 1 }} endIcon={isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}>{isExpanded ? 'Collapse' : 'Expand'}</Button></td>
                      </tr>
                      {isExpanded && (<tr><td colSpan="5" style={{ background: 'rgba(0, 27, 65, 0.02)' }}><Box sx={{ borderRadius: '13px', p: '16px', margin: '16px auto', bgcolor: '#fff', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)', maxWidth: '95%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}><Timeline prediction={result} />{result.llm_explanation && (<Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 2 }}><Typography sx={{ color: '#444', fontSize: '0.88rem', lineHeight: 1.9 }}>{result.llm_explanation}</Typography></Box>)}</Box></td></tr>)}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          )}
        </Box>
      </Box>
      
      {/* Filter Menu */}
      <Menu
        anchorEl={filterAnchorEl}
        open={Boolean(filterAnchorEl)}
        onClose={handleFilterClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'center',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'center',
        }}
        sx={{
          '& .MuiPaper-root': {
            borderRadius: '4px',
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
            width: 'auto',
            minWidth: '180px',
            maxWidth: '250px',
            mt: 0.5,
            border: '1px solid #e0e0e0'
          }
        }}
      >
        <Box sx={{ p: 1.5, borderBottom: '1px solid #e0e0e0' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, color: '#333', fontSize: '0.8rem' }}>
            Filter Options
          </Typography>
        </Box>
        <MenuItem onClick={() => handleFilterSelect('all')} sx={{ fontSize: '0.8rem', py: 1, px: 1.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <span>All</span>
            {timeFilter === 'all' && <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#001242' }} />}
          </Box>
        </MenuItem>
        <MenuItem onClick={() => handleFilterSelect('short')} sx={{ fontSize: '0.8rem', py: 1, px: 1.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <span>Short (&lt; 6 months)</span>
            {timeFilter === 'short' && <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#001242' }} />}
          </Box>
        </MenuItem>
        <MenuItem onClick={() => handleFilterSelect('medium')} sx={{ fontSize: '0.8rem', py: 1, px: 1.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <span>Medium (6-24 months)</span>
            {timeFilter === 'medium' && <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#001242' }} />}
          </Box>
        </MenuItem>
        <MenuItem onClick={() => handleFilterSelect('long')} sx={{ fontSize: '0.8rem', py: 1, px: 1.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
            <span>Long (&gt; 24 months)</span>
            {timeFilter === 'long' && <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#001242' }} />}
          </Box>
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default ResultsTableTimeSeries;
