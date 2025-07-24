import { createTheme, withTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#001B41',
      light: '#EB7836',
    },
    background: {
      default: '#f6f6f6',
      paper: '#FFFFFF',
      sidebar: '#1C2536',
      hover: '#F8FAFB'
    },
    text: {
      primary: '#333333',
      secondary: '#666666',
    },
    error: {
      main: '#dc3545',
    },
    warning: {
      main: '#ffc107',
    },
    success: {
      main: '#28a745',
    }
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      color: '#001B41',
      letterSpacing: '-0.5px',
      marginBottom: '20px',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      color: '#001B41',
      marginBottom: '20px',
    },
    h6: {
      fontSize: '1.5rem',
      fontWeight: 600,
      color: '#001B41',
    },
    body1: {
      fontSize: '1.1rem',
      color: '#666666',
      lineHeight: 1.5,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#f6f6f6',
          margin: 0,
          minHeight: '100vh'
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        contained: {
          backgroundColor: '#001B41',
          color: '#FFFFFF',
          padding: '14px',
          fontSize: '1rem',
          fontWeight: 600,
          borderRadius: '8px',
          '&:hover': {
            backgroundColor: '#EB7836',
            transform: 'translateY(-2px)',
          },
          transition: 'all 0.3s ease',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: '10px',
            '& fieldset': {
              borderColor: '#ddd',
            },
            '&:hover fieldset': {
              borderColor: '#001B41',
            },
            '& input': {
              fontSize: '1rem',
              padding: '12px',
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.05)',
          borderRadius: '16px',
          padding: '30px',
          backgroundColor: '#FFFFFF'
        },
      },
    },
    MuiListItem: {
      styleOverrides: {
        root: {
          borderRadius: '8px',
          marginBottom: '15px',
          transition: 'background-color 0.2s',
          '&:hover': {
            backgroundColor: 'rgba(244, 246, 248, 0.8)',
          },
          '&.Mui-selected': {
            backgroundColor: '#001B41',
            color: '#FFFFFF',
            '&:hover': {
              backgroundColor: '#EB7836',
            },
          },
        },
      },
    },
  },
});

export default theme; 