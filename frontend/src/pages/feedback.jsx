import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Radio, Select, MenuItem, IconButton, useMediaQuery } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import DeleteIcon from '@mui/icons-material/Delete';
import { API_BASE_URL } from '../api';

const prognoseHeaders = ['Model type', 'Model prediction', 'Your assessment', 'Comment'];
const modelOptions = ['GRU', 'XGBoost', 'TFT'];
const ratingCriteria = [
    'Clarity of the interface and layout',
    'Relevance and realism of predictions',
    'Transparency of model decisions',
    'Usefulness for daily work in Active Sourcing',
    'Trustworthiness of the AI recommendations',
    'Ease of understanding feature importance',
    'Likelihood of future usage in real scenarios',
    'Overall impression and satisfaction'
  ];

const FeedbackPage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [freeText, setFreeText] = useState('');
  const [prognoseBewertung, setPrognoseBewertung] = useState([
    { modell: '', prognose: '', echt: '', bemerkung: '' }
  ]);
  const [bewertungsskala, setBewertungsskala] = useState(Array(ratingCriteria.length).fill(3));
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handlePrognoseChange = (idx, field, value) => {
    const updated = prognoseBewertung.map((row, i) => i === idx ? { ...row, [field]: value } : row);
    setPrognoseBewertung(updated);
  };

  const addPrognoseRow = () => {
    setPrognoseBewertung([...prognoseBewertung, { modell: '', prognose: '', echt: '', bemerkung: '' }]);
  };

  const handleBewertungChange = (idx, value) => {
    const updated = [...bewertungsskala];
    updated[idx] = value;
    setBewertungsskala(updated);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(false);
    setError('');
    if (!freeText && prognoseBewertung.every(row => !row.modell && !row.prognose && !row.echt && !row.bemerkung)) {
      setError('Please fill in at least one field.');
      return;
    }
    try {
      const user = JSON.parse(localStorage.getItem('user'));
      const uid = user?.uid;
      const res = await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-User-Uid': uid },
        body: JSON.stringify({ freeText, prognoseBewertung, bewertungsskala })
      });
      if (!res.ok) throw new Error('Failed to save feedback');
      setSuccess(true);
      setFreeText('');
      setPrognoseBewertung([{ modell: '', prognose: '', echt: '', bemerkung: '' }]);
      setBewertungsskala(Array(ratingCriteria.length).fill(3));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 0, m: 0 }}>
        <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#001242', 
        mb: 2 
      }}>
        Feedback & Evaluation
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
        maxWidth: '800px' 
      }}>
        Please share your experience with the dashboard and rate the predictions and user experience. Your feedback helps us to improve the system!
      </Typography>
      <form onSubmit={handleSubmit}>
        {/* Free text feedback */}
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: 2 }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1.5, color: '#001242' }}>General feedback</Typography>
          <TextField
            label="Your feedback, suggestions, experience..."
            multiline
            minRows={4}
            fullWidth
            value={freeText}
            onChange={e => setFreeText(e.target.value)}
            sx={{ mb: 1.5, bgcolor: '#fff', borderRadius: 2 }}
          />
        </Box>
        {/* Prediction evaluation table */}
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: 2 }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1.5, color: '#001242' }}>Candidate input & prediction evaluation</Typography>
          {isMobile ? (
            <Box>
              {prognoseBewertung.map((row, idx) => (
                <Box key={idx} sx={{ mb: 3, p: 2, border: '1px solid #e0e0e0', borderRadius: 2, position: 'relative', bgcolor: '#fafbfc' }}>
                  <Box sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Model type</Typography>
                    <Select
                      value={row.modell}
                      onChange={e => handlePrognoseChange(idx, 'modell', e.target.value)}
                      displayEmpty
                      fullWidth
                      size="small"
                      sx={{ bgcolor: '#fff', borderRadius: 1 }}
                      renderValue={selected => selected || 'Select used model'}
                    >
                      <MenuItem value=""><em>Select model</em></MenuItem>
                      {modelOptions.map(opt => (
                        <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                      ))}
                    </Select>
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Model prediction</Typography>
                    <TextField value={row.prognose} onChange={e => handlePrognoseChange(idx, 'prognose', e.target.value)} size="small" placeholder="Prediction" fullWidth />
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Your assessment</Typography>
                    <TextField value={row.echt} onChange={e => handlePrognoseChange(idx, 'echt', e.target.value)} size="small" placeholder="Your assessment" fullWidth />
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Comment / Feedback</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <TextField value={row.bemerkung} onChange={e => handlePrognoseChange(idx, 'bemerkung', e.target.value)} size="small" placeholder="Comment" fullWidth multiline minRows={1} maxRows={3} />
                      {prognoseBewertung.length > 1 && (
                        <IconButton aria-label="delete row" onClick={() => {
                          setPrognoseBewertung(prognoseBewertung.filter((_, i) => i !== idx));
                        }} sx={{ ml: 1, color: 'error.main' }}>
                          <DeleteIcon />
                        </IconButton>
                      )}
                    </Box>
                  </Box>
                </Box>
              ))}
            </Box>
          ) : (
            <TableContainer component={Paper} sx={{ mb: 2, boxShadow: 0 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {prognoseHeaders.map(h => <TableCell key={h} sx={{ fontWeight: 700, color: '#001242', fontSize: '1rem', bgcolor: '#fff' }}>{h}</TableCell>)}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {prognoseBewertung.map((row, idx) => (
                    <TableRow key={idx}>
                      <TableCell sx={{ minWidth: 60, maxWidth: 80 }}>
                        <Select
                          value={row.modell}
                          onChange={e => handlePrognoseChange(idx, 'modell', e.target.value)}
                          displayEmpty
                          fullWidth
                          size="small"
                          sx={{ bgcolor: '#fff', borderRadius: 1 }}
                          renderValue={selected => selected || 'Select used model'}
                        >
                          <MenuItem value=""><em>Select model</em></MenuItem>
                          {modelOptions.map(opt => (
                            <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                          ))}
                        </Select>
                      </TableCell>
                      <TableCell sx={{ minWidth: 120, maxWidth: 180 }}>
                        <TextField value={row.prognose} onChange={e => handlePrognoseChange(idx, 'prognose', e.target.value)} size="small" placeholder="Prediction" fullWidth />
                      </TableCell>
                      <TableCell sx={{ minWidth: 120, maxWidth: 180 }}>
                        <TextField value={row.echt} onChange={e => handlePrognoseChange(idx, 'echt', e.target.value)} size="small" placeholder="Your assessment" fullWidth />
                      </TableCell>
                      <TableCell sx={{ minWidth: 320 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <TextField value={row.bemerkung} onChange={e => handlePrognoseChange(idx, 'bemerkung', e.target.value)} size="small" placeholder="Comment" fullWidth multiline minRows={1} maxRows={3} />
                          {prognoseBewertung.length > 1 && (
                            <IconButton aria-label="delete row" onClick={() => {
                              setPrognoseBewertung(prognoseBewertung.filter((_, i) => i !== idx));
                            }} sx={{ ml: 1, color: 'error.main' }}>
                              <DeleteIcon />
                            </IconButton>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          <Button onClick={addPrognoseRow} sx={{ mb: 1, color: '#EB7836', fontWeight: 700, fontSize: '1rem', textTransform: 'none' }}>+ Add row</Button>
        </Box>
        {/* Rating scale */}
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: 2 }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1.5, color: '#001242' }}>Rating scale</Typography>
          <Typography sx={{ fontSize: '0.9rem', color: '#666', mb: 1.5 }}>1 = poor, 5 = excellent</Typography>
          <TableContainer component={Paper} sx={{ mb: 1, boxShadow: 0 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, color: '#001242', fontSize: '1rem', bgcolor: '#fff' }}>Criterion</TableCell>
                  {[1,2,3,4,5].map(val => <TableCell key={val} align="center" sx={{ fontWeight: 700, color: '#001242', fontSize: '1rem', bgcolor: '#fff' }}>{val}</TableCell>)}
                </TableRow>
              </TableHead>
              <TableBody>
                {ratingCriteria.map((criterion, idx) => (
                  <TableRow key={criterion}>
                    <TableCell>{criterion}</TableCell>
                    {[1,2,3,4,5].map(val => (
                      <TableCell key={val} align="center">
                        <Radio
                          checked={bewertungsskala[idx] === val}
                          onChange={() => handleBewertungChange(idx, val)}
                          value={val}
                          name={`rating-${idx}`}
                          color="primary"
                          size="small"
                        />
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
        <Button type="submit" disabled={loading} variant="contained" sx={{ bgcolor: '#EB7836', color: '#fff', fontWeight: 700, px: 4, py: 1.2, borderRadius: 2, fontSize: '1.08rem', mb: 2, boxShadow: 2, textTransform: 'none', letterSpacing: 0.2 }}>Submit feedback</Button>
        {success && <Typography sx={{ color: 'green', mt: 2, fontWeight: 600 }}>Feedback submitted successfully!</Typography>}
        {error && <Typography sx={{ color: 'red', mt: 2 }}>{error}</Typography>}
      </form>
    </Box>
  );
};

export default FeedbackPage; 