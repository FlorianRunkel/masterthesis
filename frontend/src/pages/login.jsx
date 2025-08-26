import React, { useState, useEffect } from "react";
import { Dialog, DialogTitle, DialogContent, DialogActions, Button, Box, TextField, Typography, Paper, InputAdornment, IconButton } from '@mui/material';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import axios from 'axios';
import { API_BASE_URL } from '../api';

const Login = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [openDemoDialog, setOpenDemoDialog] = useState(false);

  useEffect(() => {
    setOpenDemoDialog(true);
  }, []);

  const handleLoginWithCredentials = async (email, password) => {
    setError('');
    if (!email || !password) {
      setError('Please enter email and password.');
      return;
    }
    try {
      const response = await axios.post(`${API_BASE_URL}/api/login`, { email, password });
      if (response.status === 200 && response.data.user) {
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('user', JSON.stringify(response.data.user));
        if (onLogin) {
          onLogin();
        }
      } else {
        setError('Wrong email or password.');
      }
    } catch (err) {
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else {
        setError('Login failed.');
      }
    }
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    await handleLoginWithCredentials(email, password);
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      bgcolor: 'linear-gradient(135deg, #fff 0%, #fff 100%)',
      background: 'linear-gradient(120deg, #fff 0%, #fff 100%)',
    }}>
      <Paper elevation={6} sx={{
        p: { xs: 3, sm: 5 },
        borderRadius: 4,
        minWidth: 340,
        maxWidth: 400,
        width: '100%',
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2
      }}>

        <Typography variant="h4" fontWeight={800} textAlign="center" color="#001242" fontSize={40}>
          Welcome Back
        </Typography>
        <Typography color="#444" textAlign="center" fontSize={15}>
          Sign into your account to get started!
        </Typography>
        <form onSubmit={handleLoginSubmit} style={{ width: '100%' }}>
          <TextField
            fullWidth
            variant="outlined"
            margin="normal"
            placeholder="Enter email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            autoComplete="email"
            sx={{
              mb: 0,
              bgcolor: '#f5f6fa',
              borderRadius: 2,
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                backgroundColor: '#f5f6fa',
                height: 48,
                fontSize: '1rem',
              },
              '& .MuiInputBase-input': {
                py: 1.5,
              },
            }}
          />
          <TextField
            fullWidth
            variant="outlined"
            margin="normal"
            placeholder="Enter password"
            type={showPassword ? 'text' : 'password'}
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
            sx={{
              mb: 2,
              bgcolor: '#f5f6fa',
              borderRadius: 2,
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
                backgroundColor: '#f5f6fa',
                height: 48,
                fontSize: '1rem',
              },
              '& .MuiInputBase-input': {
                py: 1.5,
              },
            }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton onClick={() => setShowPassword(!showPassword)} edge="end" size="small">
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
          {error && <Typography color="error" fontSize={14} mb={1}>{error}</Typography>}
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{
              mt: 1,
              bgcolor: '#EB7836',
              color: '#fff',
              fontWeight: 700,
              fontSize: '1.08rem',
              borderRadius: 2.5,
              boxShadow: '0 2px 8px #eb783664',
              letterSpacing: 0.5,
              textTransform: 'none',
              transition: '0.2s',
              '&:hover': { bgcolor: '#EB7836'}
            }}
          >
            Login
          </Button>

        </form>
        <Typography textAlign="center" fontSize={15} mt={-1.5}>
          Don't have an account yet?<br />
          Please <a href="mailto:florian.runkel@stud.uni-regensburg.de" style={{ color: '#0a1929', fontWeight: 600, textDecoration: 'underline', cursor: 'pointer' }}>reach out</a> to us!
        </Typography>
      </Paper>
    </Box>
  );
};

export default Login; 